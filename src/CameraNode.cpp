#include "ParameterHandler.hpp"
#include "format_mapping.hpp"
#include "libcamera_version_utils.hpp"
#include "pretty_print.hpp"
#include <algorithm>
#include <camera_info_manager/camera_info_manager.hpp>
#include <cassert>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#elif __has_include(<cv_bridge/cv_bridge.h>)
#include <cv_bridge/cv_bridge.h>
#endif
#include <atomic>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <future>
#include <iostream>
#include <libcamera/base/shared_fd.h>
#include <libcamera/base/signal.h>
#include <libcamera/base/span.h>
#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/control_ids.h>
#include <libcamera/formats.h>
#include <libcamera/framebuffer.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/geometry.h>
#include <libcamera/pixel_format.h>
#include <libcamera/property_ids.h>
#include <libcamera/request.h>
#include <libcamera/stream.h>
#include <memory>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <optional>
#include <rcl/context.h>
#include <rcl_interfaces/msg/floating_point_range.hpp>
#include <rcl_interfaces/msg/integer_range.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rcl_interfaces/msg/parameter_type.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/node_interfaces/node_parameters_interface.hpp>
#include <rclcpp/parameter.hpp>
#include <rclcpp/parameter_value.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/time.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <regex>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rclcpp
{
class NodeOptions;
}


namespace camera
{
class CameraNode : public rclcpp::Node
{
public:
  explicit CameraNode(const rclcpp::NodeOptions &options);

  ~CameraNode();

private:
  static libcamera::CameraManager camera_manager;
  static std::mutex camera_manager_mutex;
  std::shared_ptr<libcamera::Camera> camera;
  static std::atomic<size_t> cameras_in_use;
  libcamera::Stream *stream;
  std::shared_ptr<libcamera::FrameBufferAllocator> allocator;
  std::vector<std::unique_ptr<libcamera::Request>> requests;
  std::vector<std::thread> request_threads;
  std::unordered_map<const libcamera::Request *, std::mutex> request_mutexes;
  std::unordered_map<const libcamera::Request *, std::condition_variable> request_condvars;
  // Completion flag per request, guarded by the request's mutex. Without it, a
  // completion that arrives before process() is back in wait() is a lost
  // wakeup: that thread sleeps forever and its buffer is never re-queued,
  // until no buffers are left and the camera stops publishing.
  std::unordered_map<const libcamera::Request *, bool> request_ready;
  // Requests are processed on parallel threads and can finish out of order.
  // Publishing is serialised and only frames newer than the last one published
  // go out, so every subscriber sees monotonic stamps (H.264 decoders, such as
  // Foxglove's, reject a frame that goes back in time).
  std::mutex publish_mutex;
  int64_t last_published_stamp_ns = 0;
  std::atomic<bool> running;

  // Watchdog. libcamera can stop completing requests without reporting an
  // error: every request stays queued, no frame arrives, the node goes quiet.
  // Seen on a Pi 5 with an IMX477, both after hours of streaming and straight
  // after start-up. When no frame has arrived for watchdog_timeout seconds the
  // camera is stopped and started again in place; if that hangs, fails, or
  // still yields no frames after watchdog_restarts attempts, the process exits
  // with an error so that a supervisor (e.g. launch's respawn) can restart it.
  double watchdog_timeout = 0;  // seconds, 0 disables the watchdog
  int64_t watchdog_restarts = 3;
  std::thread watchdog_thread;
  std::mutex watchdog_mutex;
  std::condition_variable watchdog_cv;
  std::atomic<uint64_t> frame_count = 0;
  std::atomic<int64_t> last_frame_ns = 0;  // steady clock
  // Guards re-queueing against a restart: while 'restarting' is set, the
  // processing threads park their request instead of queueing it, and the
  // restart re-queues all of them once the camera has started again.
  std::mutex queue_mutex;
  std::condition_variable restart_cv;
  bool restarting = false;
  // Where each request is, guarded by queue_mutex: handed to libcamera and not
  // returned yet, or parked during a restart. Otherwise a thread is processing it.
  std::unordered_map<const libcamera::Request *, bool> request_in_camera;
  std::unordered_map<const libcamera::Request *, bool> request_parked;

  struct buffer_info_t
  {
    void *data;
    size_t size;
  };
  std::unordered_map<const libcamera::FrameBuffer *, buffer_info_t> buffer_info;

  bool use_node_time;

  static const rclcpp::PublisherOptionsWithAllocator<std::allocator<void>> pubopts;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_image;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr pub_image_compressed;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr pub_ci;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr pub_diagnostics;

  camera_info_manager::CameraInfoManager cim;

  ParameterHandler parameter_handler;
  std::string frame_id;

#ifdef RCLCPP_HAS_PARAM_EXT_CB
  // use new "post_set" callback to apply parameters
  PostSetParametersCallbackHandle::SharedPtr param_cb_change;
#else
  OnSetParametersCallbackHandle::SharedPtr param_cb_change;
#endif

  // compression quality parameter
  std::atomic_uint8_t jpeg_quality;

  void
  onDisconnect();

  void
  requestComplete(libcamera::Request *const request);

  void
  process(libcamera::Request *const request);

  void
  watchdog();

  bool
  restartCamera();

  void
  publishWatchdogDiagnostic(uint8_t level, const std::string &message);

  void
  postParameterChange(const std::vector<rclcpp::Parameter> &parameters);

#ifndef RCLCPP_HAS_PARAM_EXT_CB
  rcl_interfaces::msg::SetParametersResult
  onParameterChange(const std::vector<rclcpp::Parameter> &parameters);
#endif
};

libcamera::CameraManager CameraNode::camera_manager = {};
std::atomic<size_t> CameraNode::cameras_in_use = 0;
std::mutex camera::CameraNode::camera_manager_mutex;

RCLCPP_COMPONENTS_REGISTER_NODE(camera::CameraNode)

const rclcpp::PublisherOptionsWithAllocator<std::allocator<void>> CameraNode::pubopts = []() {
  rclcpp::PublisherOptionsWithAllocator<std::allocator<void>> options;
  options.qos_overriding_options = rclcpp::QosOverridingOptions {
    rclcpp::QosPolicyKind::Depth,
    rclcpp::QosPolicyKind::Durability,
    rclcpp::QosPolicyKind::History,
    rclcpp::QosPolicyKind::Reliability};
  return options;
}();

int64_t
steady_ns()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

libcamera::StreamRole
get_role(const std::string &role)
{
  static const std::unordered_map<std::string, libcamera::StreamRole> roles_map = {
    {"raw", libcamera::StreamRole::Raw},
    {"still", libcamera::StreamRole::StillCapture},
    {"video", libcamera::StreamRole::VideoRecording},
    {"viewfinder", libcamera::StreamRole::Viewfinder},
  };

  try {
    return roles_map.at(role);
  }
  catch (const std::out_of_range &) {
    throw std::runtime_error("invalid stream role: \"" + role + "\"");
  }
}

libcamera::Size
get_sensor_format(const std::string &format_str)
{
  if (format_str.empty()) {
    return {};
  }

  const std::regex pattern(R"((\d+):(\d+))");
  std::smatch match;

  if (std::regex_match(format_str, match, pattern)) {
    try {
      const int width = std::stoi(match[1].str());
      const int height = std::stoi(match[2].str());

      return libcamera::Size {static_cast<unsigned int>(width), static_cast<unsigned int>(height)};
    }
    catch (const std::out_of_range &) {
      throw std::runtime_error("Invalid sensor_mode. Width or height exceeds maximum value of " +
                               std::to_string(std::numeric_limits<int>::max()));
    }
    catch (const std::invalid_argument &) {
      // Unexpected - throw exception below
    }
  }

  // Throw exception as it was not possible to parse the format string
  throw std::runtime_error("Invalid sensor_mode. Expected [width]:[height] but got " + format_str);
}

// The following function "compressImageMsg" is adapted from "CvImage::toCompressedImageMsg"
// (https://github.com/ros-perception/vision_opencv/blob/066793a23e5d06d76c78ca3d69824a501c3554fd/cv_bridge/src/cv_bridge.cpp#L512-L535)
// and covered by the BSD-3-Clause licence.
void
compressImageMsg(const sensor_msgs::msg::Image &source,
                 sensor_msgs::msg::CompressedImage &destination,
                 const std::vector<int> &params = std::vector<int>())
{
  namespace enc = sensor_msgs::image_encodings;

  std::shared_ptr<CameraNode> tracked_object;
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(source, tracked_object);

  destination.header = source.header;
  cv::Mat image;
  if (cv_ptr->encoding == enc::BGR8 || cv_ptr->encoding == enc::BGRA8 ||
      cv_ptr->encoding == enc::MONO8 || cv_ptr->encoding == enc::MONO16) {
    image = cv_ptr->image;
  }
  else {
    cv_bridge::CvImagePtr tempThis = std::make_shared<cv_bridge::CvImage>(*cv_ptr);
    cv_bridge::CvImagePtr temp;
    if (enc::hasAlpha(cv_ptr->encoding)) {
      temp = cv_bridge::cvtColor(tempThis, enc::BGRA8);
    }
    else {
      temp = cv_bridge::cvtColor(tempThis, enc::BGR8);
    }
    image = temp->image;
  }

  destination.format = "jpg";
  cv::imencode(".jpg", image, destination.data, params);
}


CameraNode::CameraNode(const rclcpp::NodeOptions &options)
    : Node("camera", options),
#if CIM_HAS_NODE_INTERFACE
      cim(
        this->get_node_base_interface(),
        this->get_node_services_interface(),
        this->get_node_logging_interface()
#if CIM_HAS_QoS
          ,
        "camera", {}, rclcpp::SystemDefaultsQoS()
#endif
          ),
#else
      cim(this),
#endif
      parameter_handler(this),
      param_cb_change(
#ifdef RCLCPP_HAS_PARAM_EXT_CB
        add_post_set_parameters_callback(std::bind(&CameraNode::postParameterChange, this, std::placeholders::_1))
#else
        add_on_set_parameters_callback(std::bind(&CameraNode::onParameterChange, this, std::placeholders::_1))
#endif
      )
{
  // pixel format
  rcl_interfaces::msg::ParameterDescriptor param_descr_format;
  param_descr_format.description = "pixel format of streaming buffer";
  param_descr_format.read_only = true;
  const std::string &format = declare_parameter<std::string>("format", {}, param_descr_format);

  // stream role
  rcl_interfaces::msg::ParameterDescriptor param_descr_role;
  param_descr_role.description = "stream role";
  param_descr_role.additional_constraints = "one of {raw, still, video, viewfinder}";
  param_descr_role.read_only = true;
  const libcamera::StreamRole role =
    get_role(declare_parameter<std::string>("role", "viewfinder", param_descr_role));

  // image dimensions
  rcl_interfaces::msg::ParameterDescriptor param_descr_ro;
  param_descr_ro.read_only = true;
  const uint32_t w = declare_parameter<int64_t>("width", {}, param_descr_ro);
  const uint32_t h = declare_parameter<int64_t>("height", {}, param_descr_ro);
  const libcamera::Size size {w, h};

  // Raw format dimensions
  rcl_interfaces::msg::ParameterDescriptor param_descr_sensor_mode;
  param_descr_sensor_mode.description = "raw mode of the sensor";
  param_descr_sensor_mode.additional_constraints = "string in format [width]:[height]";
  param_descr_sensor_mode.read_only = true;
  const libcamera::Size sensor_size = get_sensor_format(declare_parameter<std::string>("sensor_mode", {}, param_descr_sensor_mode));

  // camera frame_id
  frame_id = declare_parameter<std::string>("frame_id", "camera", param_descr_ro);

  rcl_interfaces::msg::ParameterDescriptor param_descr_orientation;
  param_descr_orientation.description = "camera orientation";
  rcl_interfaces::msg::IntegerRange orientation_range;
  orientation_range.from_value = 0;
  orientation_range.to_value = 270;
  orientation_range.step = 90;
  param_descr_orientation.integer_range.push_back(orientation_range);
  param_descr_orientation.read_only = true;
  constexpr int orientation_angle_default = 0;
  const int angle = declare_parameter<int>("orientation", orientation_angle_default, param_descr_orientation);
#if LIBCAMERA_VER_GE(0, 2, 0)
  const libcamera::Orientation orientation = libcamera::orientationFromRotation(angle);
#else
  if (angle != orientation_angle_default) {
    RCLCPP_WARN_STREAM(get_logger(), "parameter 'orientation' not supported on libcamera " << LIBCAMERA_VERSION_MAJOR << "." << LIBCAMERA_VERSION_MINOR);
  }
#endif

  // camera info file url
  rcl_interfaces::msg::ParameterDescriptor param_descr_camera_info_url;
  param_descr_camera_info_url.description = "camera calibration info file url";
  param_descr_camera_info_url.read_only = true;

  // camera ID
  const rclcpp::ParameterValue &camera_id =
    declare_parameter("camera", rclcpp::ParameterValue {}, param_descr_ro.set__dynamic_typing(true));

  // we cannot control the compression rate of the libcamera MJPEG stream
  // ignore "jpeg_quality" parameter for MJPEG streams
  if (libcamera::PixelFormat::fromString(format) != libcamera::formats::MJPEG) {
    rcl_interfaces::msg::ParameterDescriptor jpeg_quality_description;
    jpeg_quality_description.name = "jpeg_quality";
    jpeg_quality_description.type = rcl_interfaces::msg::ParameterType::PARAMETER_INTEGER;
    jpeg_quality_description.description = "Image quality for JPEG format";
    jpeg_quality_description.read_only = false;
    rcl_interfaces::msg::IntegerRange jpeg_range;
    jpeg_range.from_value = 1;
    jpeg_range.to_value = 100;
    jpeg_range.step = 1;
    jpeg_quality_description.integer_range = {jpeg_range};
    // default to 95
    jpeg_quality = declare_parameter<uint8_t>("jpeg_quality", 95, jpeg_quality_description);
  }

  // use_node_time parameter
  rcl_interfaces::msg::ParameterDescriptor param_descr_use_node_time;
  param_descr_use_node_time.description = "use node time instead of sensor timestamp for image messages";
  param_descr_use_node_time.read_only = true;
  use_node_time = declare_parameter<bool>("use_node_time", false, param_descr_use_node_time);

  // watchdog
  rcl_interfaces::msg::ParameterDescriptor param_descr_watchdog_timeout;
  param_descr_watchdog_timeout.description =
    "restart the camera when no frame has arrived for this many seconds (0 disables the watchdog)";
  param_descr_watchdog_timeout.read_only = true;
  watchdog_timeout = declare_parameter<double>("watchdog_timeout", 0.0, param_descr_watchdog_timeout);
  if (watchdog_timeout < 0)
    throw std::runtime_error("watchdog_timeout must not be negative");

  rcl_interfaces::msg::ParameterDescriptor param_descr_watchdog_restarts;
  param_descr_watchdog_restarts.description =
    "in-place camera restarts the watchdog tries before exiting the process with an error";
  param_descr_watchdog_restarts.read_only = true;
  watchdog_restarts = declare_parameter<int64_t>("watchdog_restarts", 3, param_descr_watchdog_restarts);
  if (watchdog_restarts < 0)
    throw std::runtime_error("watchdog_restarts must not be negative");

  // publisher for raw and compressed image
  pub_image = this->create_publisher<sensor_msgs::msg::Image>("~/image_raw", 1, pubopts);
  pub_image_compressed =
    this->create_publisher<sensor_msgs::msg::CompressedImage>("~/image_raw/compressed", 1, pubopts);
  pub_ci = this->create_publisher<sensor_msgs::msg::CameraInfo>("~/camera_info", 1, pubopts);
  pub_diagnostics =
    this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/diagnostics", 1);

  // start camera manager and check for cameras
  std::scoped_lock camera_manager_lk(camera_manager_mutex);
  const int ec_start = camera_manager.start();
  if (ec_start < 0)
    throw std::runtime_error("failed to start camera manager: " + std::string(std::strerror(-ec_start)));
  if (camera_manager.cameras().empty())
    throw std::runtime_error("no cameras available");

  // get the camera
  switch (camera_id.get_type()) {
  case rclcpp::ParameterType::PARAMETER_NOT_SET:
    // use first camera as default
    camera = camera_manager.cameras().front();
    RCLCPP_INFO_STREAM(get_logger(), camera_manager);
    RCLCPP_WARN_STREAM(get_logger(),
                       "no camera selected, using default: \"" << camera->id() << "\"");
    RCLCPP_WARN_STREAM(get_logger(), "set parameter 'camera' to silence this warning");
    break;
  case rclcpp::ParameterType::PARAMETER_INTEGER:
  {
    const size_t &id = camera_id.get<rclcpp::ParameterType::PARAMETER_INTEGER>();
    if (id >= camera_manager.cameras().size()) {
      RCLCPP_INFO_STREAM(get_logger(), camera_manager);
      throw std::runtime_error("camera with id " + std::to_string(id) + " does not exist");
    }
    camera = camera_manager.cameras().at(id);
    RCLCPP_DEBUG_STREAM(get_logger(), "found camera by id: " << id);
  } break;
  case rclcpp::ParameterType::PARAMETER_STRING:
  {
    const std::string &name = camera_id.get<rclcpp::ParameterType::PARAMETER_STRING>();
    camera = camera_manager.get(name);
    if (!camera) {
      RCLCPP_INFO_STREAM(get_logger(), camera_manager);
      throw std::runtime_error("camera with name " + name + " does not exist");
    }
    RCLCPP_DEBUG_STREAM(get_logger(), "found camera by name: \"" << name << "\"");
  } break;
  default:
    RCLCPP_FATAL_STREAM(get_logger(), "unsupported camera parameter type: "
                                        << rclcpp::to_string(camera_id.get_type()));
    break;
  }

  if (!camera)
    throw std::runtime_error("failed to find camera");

  switch (camera->acquire()) {
  case 0:
    // OK
    break;
  case -ENODEV:
    throw std::runtime_error("failed to acquire camera: The camera has been disconnected from the system");
  case -EBUSY:
    throw std::runtime_error("failed to acquire camera: The camera is not free and can't be acquired by the caller");
  default:
    throw std::runtime_error("failed to acquire camera: unknown status");
  }

  camera->disconnected.connect(this, &CameraNode::onDisconnect);

  std::vector<libcamera::StreamRole> roles {role};

  // Add the RAW role if the sensor_size is defined
  if (!sensor_size.isNull() && role != libcamera::StreamRole::Raw) {
    roles.push_back(libcamera::StreamRole::Raw);
  }

  // configure camera stream
  std::unique_ptr<libcamera::CameraConfiguration> cfg =
    camera->generateConfiguration(roles);

  if (!cfg || cfg->size() != roles.size())
    throw std::runtime_error("failed to generate configuration for all roles");

#if LIBCAMERA_VER_GE(0, 2, 0)
  cfg->orientation = orientation;
#endif

  libcamera::StreamConfiguration &scfg = cfg->at(0);

  // list all camera formats, including those not supported by the ROS message
  RCLCPP_DEBUG_STREAM(get_logger(), "default " << role << " stream configuration: \"" << scfg.toString() << "\"");
  RCLCPP_DEBUG_STREAM(get_logger(), scfg.formats());

  // get common pixel formats that are supported by the camera and the node
  const libcamera::StreamFormats stream_formats = get_common_stream_formats(scfg.formats());
  const std::vector<libcamera::PixelFormat> common_fmt = stream_formats.pixelformats();

  RCLCPP_INFO_STREAM(get_logger(), stream_formats);

  if (common_fmt.empty())
    throw std::runtime_error("camera does not provide any of the supported pixel formats");

  if (format.empty()) {
    // check that the default pixel format is supported by the ROS encoding
    if (std::find(common_fmt.cbegin(), common_fmt.cend(), scfg.pixelFormat) == common_fmt.cend()) {
      // auto select first common pixel format
      RCLCPP_WARN_STREAM(get_logger(), "default pixel format (" << scfg.pixelFormat << ") not supported");
      scfg.pixelFormat = common_fmt.front();
    }

    RCLCPP_WARN_STREAM(get_logger(),
                       "no pixel format selected, auto-selecting: \"" << scfg.pixelFormat << "\"");
    RCLCPP_WARN_STREAM(get_logger(), "set parameter 'format' to silence this warning");
  }
  else {
    // get pixel format from provided string
    const libcamera::PixelFormat format_requested = libcamera::PixelFormat::fromString(format);
    if (!format_requested.isValid()) {
      throw std::runtime_error("invalid pixel format: \"" + format + "\"");
    }
    // check that the requested format is supported by camera and the node
    if (std::find(common_fmt.begin(), common_fmt.end(), format_requested) == common_fmt.end()) {
      throw std::runtime_error("unsupported pixel format \"" + format + "\"");
    }
    scfg.pixelFormat = format_requested;
  }

  RCLCPP_INFO_STREAM(get_logger(), list_format_sizes(scfg));

  if (size.isNull()) {
    RCLCPP_WARN_STREAM(get_logger(), "no dimensions selected, auto-selecting: \"" << scfg.size << "\"");
    RCLCPP_WARN_STREAM(get_logger(), "set parameter 'width' or 'height' to silence this warning");
  }
  else {
    scfg.size = size;
  }

  if (!sensor_size.isNull() && role != libcamera::StreamRole::Raw) {
    libcamera::StreamConfiguration &modecfg = cfg->at(1);
    modecfg.size = sensor_size;
    RCLCPP_INFO_STREAM(get_logger(), "Sensor mode configuration: " << modecfg.toString());
  }

  // store selected stream configuration
  const libcamera::StreamConfiguration selected_scfg = scfg;

  switch (cfg->validate()) {
  case libcamera::CameraConfiguration::Valid:
    break;
  case libcamera::CameraConfiguration::Adjusted:
#if LIBCAMERA_VER_GE(0, 2, 0)
    RCLCPP_WARN_STREAM(get_logger(), "stream configuration adjusted from \""
                                       << selected_scfg.toString() << "\" (" << orientation << ") to \"" << scfg.toString()
                                       << "\" (" << cfg->orientation << ")");
    if (cfg->orientation != orientation) {
      RCLCPP_WARN_STREAM(get_logger(), "cannot set orientation to " << orientation);
    }
#else
    RCLCPP_WARN_STREAM(get_logger(), "stream configuration adjusted from \""
                                       << selected_scfg.toString() << "\" to \"" << scfg.toString()
                                       << "\"");
#endif
    break;
  case libcamera::CameraConfiguration::Invalid:
    throw std::runtime_error("failed to validate stream configurations");
    break;
  }

  switch (camera->configure(cfg.get())) {
  case -ENODEV:
    throw std::runtime_error("configure: camera has been disconnected from the system");
  case -EACCES:
    throw std::runtime_error("configure: camera is not in a state where it can be configured");
  case -EINVAL:
    throw std::runtime_error("configure: configuration \"" + scfg.toString() + "\" is not valid");
  default:
    RCLCPP_INFO_STREAM(get_logger(), "camera \"" << camera->id() << "\" configured with "
                                                 << scfg.toString() << " stream");
    break;
  }

  // format camera name for calibration file
  const libcamera::ControlList &props = camera->properties();
  std::string cname = camera->id() + '_' + scfg.size.toString();
  if (const std::optional<std::string_view> model = props.get(libcamera::properties::Model)) {
    cname = static_cast<std::string>(model.value()) + '_' + cname;
  }
  if (!sensor_size.isNull() && role != libcamera::StreamRole::Raw)
    cname = cname + '_' + cfg->at(1).toString();

  // clean camera name of non-alphanumeric characters
  cname.erase(
    std::remove_if(cname.begin(), cname.end(), [](const char &x) { return std::isspace(x); }),
    cname.cend());
  std::replace_if(
    cname.begin(), cname.end(), [](const char &x) { return !std::isalnum(x); }, '_');

  if (!cim.setCameraName(cname))
    throw std::runtime_error("camera name must only contain alphanumeric characters");

  const std::string &camera_info_url = declare_parameter<std::string>(
    "camera_info_url", {}, param_descr_camera_info_url);
  if (!cim.loadCameraInfo(camera_info_url)) {
    if (!camera_info_url.empty()) {
      RCLCPP_WARN_STREAM(get_logger(), "failed to load camera calibration info from provided URL, using default URL");
      cim.loadCameraInfo({});
    }
  }

  parameter_handler.declare(camera->controls());

  // allocate stream buffers and create one request per buffer
  stream = scfg.stream();

  allocator = std::make_shared<libcamera::FrameBufferAllocator>(camera);
  const int nbuffer = allocator->allocate(stream);

  if (nbuffer < 0) {
    throw std::runtime_error("allocation failed: " + std::string(std::strerror(-nbuffer)));
  }

  for (const std::unique_ptr<libcamera::FrameBuffer> &buffer : allocator->buffers(stream)) {
    std::unique_ptr<libcamera::Request> request = camera->createRequest();
    if (!request)
      throw std::runtime_error("Can't create request");

    // multiple planes of the same buffer use the same file descriptor
    size_t buffer_length = 0;
    int fd = -1;
    for (const libcamera::FrameBuffer::Plane &plane : buffer->planes()) {
      if (plane.offset == libcamera::FrameBuffer::Plane::kInvalidOffset)
        throw std::runtime_error("invalid offset");
      buffer_length = std::max<size_t>(buffer_length, plane.offset + plane.length);
      if (!plane.fd.isValid())
        throw std::runtime_error("file descriptor is not valid");
      if (fd == -1)
        fd = plane.fd.get();
      else if (fd != plane.fd.get())
        throw std::runtime_error("plane file descriptors differ");
    }

    // memory-map the frame buffer planes
    void *data = mmap(nullptr, buffer_length, PROT_READ, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED)
      throw std::runtime_error("mmap failed: " + std::string(std::strerror(errno)));
    buffer_info[buffer.get()] = {data, buffer_length};

    if (request->addBuffer(stream, buffer.get()) < 0)
      throw std::runtime_error("Can't set buffer for request");

    requests.push_back(std::move(request));
  }

  assert(nbuffer >= 0 && requests.size() == size_t(nbuffer));

  // create a processing thread per request
  running = true;
  for (const std::unique_ptr<libcamera::Request> &request : requests) {
    // create mutexes in-place
    request_mutexes[request.get()];
    request_condvars[request.get()];
    request_ready[request.get()] = false;
    request_in_camera[request.get()] = false;
    request_parked[request.get()] = false;
    request_threads.emplace_back(&CameraNode::process, this, request.get());
  }

  // register callback
  camera->requestCompleted.connect(this, &CameraNode::requestComplete);

  // start camera with initial controls
  switch (camera->start(&parameter_handler.get_control_values())) {
  case 0:
    // OK
    break;
  case -ENODEV:
    throw std::runtime_error("failed to start camera: The camera has been disconnected from the system");
  case -EBUSY:
    throw std::runtime_error("failed to start camera: The camera is not in a state where it can be started");
  default:
    throw std::runtime_error("failed to start camera: unknown status");
  }

  cameras_in_use++;

  // queue all requests
  {
    std::lock_guard queue_lk(queue_mutex);
    for (std::unique_ptr<libcamera::Request> &request : requests) {
      if (const int ret = camera->queueRequest(request.get()); ret < 0) {
        RCLCPP_WARN_STREAM(get_logger(), "failed to queue request (" << request->toString() << "): " << strerror(-ret));
      }
      else {
        request_in_camera.at(request.get()) = true;
      }
    }
  }

  if (watchdog_timeout > 0) {
    last_frame_ns = steady_ns();
    watchdog_thread = std::thread(&CameraNode::watchdog, this);
    RCLCPP_INFO_STREAM(get_logger(), "watchdog enabled: restart after " << watchdog_timeout
                                                                       << " s without a frame, up to "
                                                                       << watchdog_restarts << " times");
  }
}

CameraNode::~CameraNode()
{
  camera->disconnected.disconnect(this, &CameraNode::onDisconnect);

  // stop request callbacks
  camera->requestCompleted.disconnect(this, &CameraNode::requestComplete);

  // stop request processing threads
  running = false;

  // unlock all threads
  for (auto &[req, condvar] : request_condvars)
    condvar.notify_all();

  // stop the watchdog, and any restart it is waiting on (taking each mutex
  // before notifying, so a thread between its check and its wait can't miss it)
  {
    std::lock_guard lk(queue_mutex);
  }
  restart_cv.notify_all();
  {
    std::lock_guard lk(watchdog_mutex);
  }
  watchdog_cv.notify_all();
  if (watchdog_thread.joinable())
    watchdog_thread.join();

  // wait for all currently running threads to finish
  for (std::thread &thread : request_threads)
    thread.join();

  // stop camera
  if (camera->stop()) {
    RCLCPP_ERROR_STREAM(get_logger(), "failed to stop camera");
  }
  const int ec_alloc_free = allocator->free(stream);
  if (ec_alloc_free < 0) {
    RCLCPP_ERROR_STREAM(get_logger(), "failed to free buffers: " << std::strerror(-ec_alloc_free));
  }
  allocator.reset();
  switch (camera->release()) {
  case 0:
    // OK
    break;
  case -EBUSY:
    RCLCPP_ERROR_STREAM(get_logger(), "failed to release camera: The camera is running and can't be released");
    break;
  default:
    RCLCPP_ERROR_STREAM(get_logger(), "failed to release camera: unknown status");
  }
  camera.reset();
  camera_manager_mutex.lock();
  cameras_in_use--;
  if (cameras_in_use == 0) {
    camera_manager.stop();
  }
  camera_manager_mutex.unlock();
  for (const auto &e : buffer_info)
    if (munmap(e.second.data, e.second.size) == -1)
      std::cerr << "munmap failed: " << std::strerror(errno) << std::endl;
}

void
CameraNode::onDisconnect()
{
  RCLCPP_FATAL_STREAM(get_logger(), "camera '" << camera->id() << "' disconnected!");

  if (pub_diagnostics->get_subscription_count()) {
    diagnostic_msgs::msg::DiagnosticArray diagnostic_array;
    diagnostic_array.header.stamp = this->now();
    diagnostic_array.header.frame_id = frame_id;

    diagnostic_msgs::msg::DiagnosticStatus diagnostic_status;
    diagnostic_status.hardware_id = camera->id();
    diagnostic_status.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
    diagnostic_status.message = "camera disconnected";

    diagnostic_array.status.push_back(diagnostic_status);
    pub_diagnostics->publish(diagnostic_array);
  }

  running = false;
  for (auto &[req, condvar] : request_condvars)
    condvar.notify_all();
}

void
CameraNode::requestComplete(libcamera::Request *const request)
{
  {
    std::lock_guard queue_lk(queue_mutex);
    request_in_camera.at(request) = false;
  }
  {
    std::unique_lock lk(request_mutexes.at(request));
    request_ready.at(request) = true;
  }
  request_condvars.at(request).notify_one();
}

void
CameraNode::process(libcamera::Request *const request)
{
  while (true) {
    // block until request is available (predicate: no lost or spurious wakeups)
    std::unique_lock lk(request_mutexes.at(request));
    request_condvars.at(request).wait(lk, [&] { return request_ready.at(request) || !running; });

    if (!running)
      return;

    request_ready.at(request) = false;
    // Don't hold the lock while processing: requestComplete() runs on
    // libcamera's event thread and would block it behind this frame's work.
    lk.unlock();

    // prepare message header
    std_msgs::msg::Header hdr;
    hdr.frame_id = frame_id;

    // if using sensor timestamps, get the sensor timestamp from the request metadata
    int64_t sensor_latency = 0;
    if (!use_node_time) {
      const libcamera::ControlList &req_metadata = request->metadata();
      if (const std::optional<int64_t> sensor_ts = req_metadata.get(libcamera::controls::SensorTimestamp)) {
        sensor_latency = rclcpp::Clock(RCL_STEADY_TIME).now().nanoseconds() - sensor_ts.value();
      }
      else {
        RCLCPP_WARN_STREAM_ONCE(get_logger(), "sensor timestamp not available, falling back to node time as reference");
      }
    }

    // Adjust timestamp by the sensor latency
    hdr.stamp = this->now() - rclcpp::Duration::from_nanoseconds(sensor_latency);

    diagnostic_msgs::msg::DiagnosticArray diagnostic_array;
    diagnostic_array.header = hdr;

    diagnostic_msgs::msg::DiagnosticStatus diagnostic_status;
    diagnostic_status.hardware_id = camera->id();

    if (request->status() == libcamera::Request::RequestComplete) {
      assert(request->buffers().size() == 1);

      diagnostic_status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;

      last_frame_ns = steady_ns();
      frame_count++;

      // publish metadata, sorted by control ID
      for (const auto &[id, value] : std::map {
             request->metadata().begin(),
             request->metadata().end(),
           }) {
        diagnostic_msgs::msg::KeyValue kv;
        kv.key = libcamera::controls::controls.at(id)->name();
        kv.value = value.toString();
        diagnostic_status.values.push_back(kv);
      }

      // get the stream and buffer from the request
      const libcamera::FrameBuffer *buffer = request->findBuffer(stream);
      const libcamera::FrameMetadata &metadata = buffer->metadata();
      size_t bytesused = 0;
      for (const libcamera::FrameMetadata::Plane &plane : metadata.planes())
        bytesused += plane.bytesused;

      // prepare image messages
      const libcamera::StreamConfiguration &cfg = stream->configuration();

      auto msg_img = std::make_unique<sensor_msgs::msg::Image>();
      auto msg_img_compressed = std::make_unique<sensor_msgs::msg::CompressedImage>();

      if (format_type(cfg.pixelFormat) == FormatType::RAW) {
        // raw uncompressed image
        assert(buffer_info[buffer].size == bytesused);
        msg_img->header = hdr;
        msg_img->width = cfg.size.width;
        msg_img->height = cfg.size.height;
        msg_img->step = cfg.stride;
        msg_img->encoding = get_ros_encoding(cfg.pixelFormat);
        msg_img->is_bigendian = (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__);
        msg_img->data.resize(buffer_info[buffer].size);
        memcpy(msg_img->data.data(), buffer_info[buffer].data, buffer_info[buffer].size);

        // compress to jpeg
        if (pub_image_compressed->get_subscription_count()) {
          try {
            compressImageMsg(*msg_img, *msg_img_compressed,
                             {cv::IMWRITE_JPEG_QUALITY, jpeg_quality});
          }
          catch (const cv_bridge::Exception &e) {
            RCLCPP_ERROR_STREAM(get_logger(), e.what());
          }
        }
      }
      else if (format_type(cfg.pixelFormat) == FormatType::COMPRESSED) {
        // compressed image
        assert(bytesused < buffer_info[buffer].size);
        msg_img_compressed->header = hdr;
        msg_img_compressed->format = get_ros_encoding(cfg.pixelFormat);
        msg_img_compressed->data.resize(bytesused);
        memcpy(msg_img_compressed->data.data(), buffer_info[buffer].data, bytesused);

        // decompress into raw rgb8 image
        if (pub_image->get_subscription_count())
          cv_bridge::toCvCopy(*msg_img_compressed, "rgb8")->toImageMsg(*msg_img);
      }
      else {
        throw std::runtime_error("unsupported pixel format: " +
                                 stream->configuration().pixelFormat.toString());
      }

      {
        // Publish in stamp order (see publish_mutex). The check and the
        // publishes share one lock, or two threads could each pass the check
        // and still publish in the wrong order.
        std::lock_guard<std::mutex> publish_lk(publish_mutex);
        const int64_t stamp_ns = rclcpp::Time(hdr.stamp).nanoseconds();
        if (stamp_ns > last_published_stamp_ns) {
          last_published_stamp_ns = stamp_ns;

          pub_image->publish(std::move(msg_img));
          pub_image_compressed->publish(std::move(msg_img_compressed));

          sensor_msgs::msg::CameraInfo ci = cim.getCameraInfo();
          ci.header = hdr;
          pub_ci->publish(ci);
        }
        else {
          RCLCPP_DEBUG_STREAM(get_logger(), "dropped out-of-order frame ("
                                              << (stamp_ns - last_published_stamp_ns) / 1000 << " us behind)");
        }
      }
    }
    else if (request->status() == libcamera::Request::RequestCancelled) {
      diagnostic_status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
      bool expected;
      {
        std::lock_guard queue_lk(queue_mutex);
        expected = restarting;  // the watchdog's camera->stop() cancels every queued request
      }
      if (expected)
        RCLCPP_DEBUG_STREAM(get_logger(), "request '" << request->toString() << "' cancelled by restart");
      else
        RCLCPP_ERROR_STREAM(get_logger(), "request '" << request->toString() << "' cancelled");
    }

    diagnostic_array.status.push_back(diagnostic_status);

    pub_diagnostics->publish(diagnostic_array);

    // redeclare implicitly undeclared parameters
    parameter_handler.redeclare();

    // queue the request again for the next frame and update controls
    request->reuse(libcamera::Request::ReuseBuffers);
    parameter_handler.move_control_values(request->controls());

    for (const auto &[id, value] : request->controls()) {
      const std::string &name = libcamera::controls::controls.at(id)->name();
      RCLCPP_DEBUG_STREAM(get_logger(), "applied control '" << name << "': " << (value.isNone() ? "NONE" : value.toString()));
    }

    {
      std::lock_guard queue_lk(queue_mutex);
      if (restarting) {
        // The camera is being restarted and cannot take requests; hand this
        // one to restartCamera(), which queues it once the camera is running.
        request_parked.at(request) = true;
        restart_cv.notify_all();
        continue;
      }
      if (const int ret = camera->queueRequest(request); ret < 0) {
        RCLCPP_WARN_STREAM(get_logger(), "failed to queue request (" << request->toString() << "): " << strerror(-ret));
        if (ret == -ENODEV || ret == -EACCES) {
          running = false;
          return;
        }
      }
      else {
        request_in_camera.at(request) = true;
      }
    }

    if (!parameter_handler.sync_control_values(request->controls())) {
      RCLCPP_WARN_STREAM(get_logger(), "Failed to synchronise control values of queued request!");
    }
  }
}

void
CameraNode::publishWatchdogDiagnostic(const uint8_t level, const std::string &message)
{
  diagnostic_msgs::msg::DiagnosticArray diagnostic_array;
  diagnostic_array.header.stamp = this->now();
  diagnostic_array.header.frame_id = frame_id;

  diagnostic_msgs::msg::DiagnosticStatus diagnostic_status;
  diagnostic_status.hardware_id = camera->id();
  diagnostic_status.level = level;
  diagnostic_status.message = message;

  diagnostic_array.status.push_back(diagnostic_status);
  pub_diagnostics->publish(diagnostic_array);
}

void
CameraNode::watchdog()
{
  using namespace std::chrono;
  const auto timeout = duration_cast<nanoseconds>(duration<double>(watchdog_timeout));
  // check a few times per timeout, so a stall is caught within ~1.25 timeouts
  const auto period = timeout / 4;
  // a restart that takes longer than this is treated as hung
  const auto restart_deadline = std::max<nanoseconds>(seconds(10), 2 * timeout);

  int64_t attempts = 0;
  uint64_t frames_at_restart = frame_count;
  int64_t restarted_ns = 0;

  // Leave the process without running destructors: they would stop the
  // camera, which is what just hung or failed.
  const auto give_up = [this](const std::string &why) {
    RCLCPP_FATAL_STREAM(get_logger(), "watchdog: " << why << ", exiting so the node can be restarted");
    publishWatchdogDiagnostic(diagnostic_msgs::msg::DiagnosticStatus::ERROR, "watchdog: " + why);
    std::cout << std::flush;
    std::cerr << std::flush;
    std::_Exit(EXIT_FAILURE);
  };

  std::unique_lock lk(watchdog_mutex);
  while (running) {
    watchdog_cv.wait_for(lk, period, [this] { return !running; });
    if (!running)
      return;

    if (attempts > 0 && frame_count != frames_at_restart) {
      RCLCPP_INFO_STREAM(get_logger(), "watchdog: frames resumed after restart");
      publishWatchdogDiagnostic(diagnostic_msgs::msg::DiagnosticStatus::OK, "watchdog: frames resumed after restart");
      attempts = 0;
    }

    const int64_t silent_ns = steady_ns() - std::max<int64_t>(last_frame_ns, restarted_ns);
    if (silent_ns < timeout.count())
      continue;

    const std::string silence = "no frame for " + std::to_string(silent_ns / 1000000) + " ms";
    if (attempts >= watchdog_restarts)
      give_up(silence + " after " + std::to_string(attempts) + " restart(s)");

    attempts++;
    RCLCPP_WARN_STREAM(get_logger(), "watchdog: " << silence << ", restarting camera (attempt "
                                                  << attempts << "/" << watchdog_restarts << ")");
    publishWatchdogDiagnostic(diagnostic_msgs::msg::DiagnosticStatus::WARN,
                              "watchdog: " + silence + ", restarting camera");

    // Run the restart on its own thread: if libcamera is wedged, stop() may
    // never return, and then the only way out is to exit the process.
    auto result = std::make_shared<std::promise<bool>>();
    std::future<bool> done = result->get_future();
    lk.unlock();
    std::thread([this, result] { result->set_value(restartCamera()); }).detach();
    const bool finished = done.wait_for(restart_deadline) == std::future_status::ready;
    lk.lock();

    if (!finished)
      give_up("camera restart hung");
    if (!running)
      return;  // node is shutting down
    if (!done.get())
      give_up("camera restart failed");

    frames_at_restart = frame_count;
    restarted_ns = steady_ns();
  }
}

bool
CameraNode::restartCamera()
{
  {
    std::lock_guard lk(queue_mutex);
    restarting = true;
    for (auto &[request, parked] : request_parked)
      parked = false;
  }

  // stop() cancels every request libcamera still holds and delivers those
  // completions before it returns; each processing thread then parks its
  // request instead of re-queueing it
  if (const int ret = camera->stop(); ret < 0) {
    RCLCPP_ERROR_STREAM(get_logger(), "watchdog: failed to stop camera: " << strerror(-ret));
    return false;
  }

  {
    std::unique_lock lk(queue_mutex);
    // A request still marked as in the camera now was never returned: libcamera
    // lost it, which is one way the camera can go silent. Wait for the others.
    const auto settled = [this] {
      return std::all_of(requests.cbegin(), requests.cend(), [this](const auto &request) {
        return request_parked.at(request.get()) || request_in_camera.at(request.get());
      });
    };
    const bool all_settled = restart_cv.wait_for(lk, std::chrono::seconds(5), [&] {
      return settled() || !running;
    });
    if (!running)
      return false;
    if (!all_settled) {
      const size_t busy = std::count_if(requests.cbegin(), requests.cend(), [this](const auto &request) {
        return !request_parked.at(request.get()) && !request_in_camera.at(request.get());
      });
      RCLCPP_ERROR_STREAM(get_logger(), "watchdog: " << busy << " of " << requests.size()
                                                     << " requests still being processed 5 s after stopping the camera");
      return false;
    }

    // reclaim requests libcamera never returned; their threads are idle,
    // waiting for a completion, and will process them once re-queued
    size_t lost = 0;
    for (const std::unique_ptr<libcamera::Request> &request : requests) {
      if (request_in_camera.at(request.get())) {
        request_in_camera.at(request.get()) = false;
        request->reuse(libcamera::Request::ReuseBuffers);
        lost++;
      }
    }
    if (lost) {
      RCLCPP_WARN_STREAM(get_logger(), "watchdog: libcamera never returned " << lost << " of " << requests.size()
                                                                           << " requests, reclaiming them");
    }
  }

  if (const int ret = camera->start(&parameter_handler.get_control_values()); ret < 0) {
    RCLCPP_ERROR_STREAM(get_logger(), "watchdog: failed to start camera: " << strerror(-ret));
    return false;
  }

  std::lock_guard lk(queue_mutex);
  restarting = false;
  for (const std::unique_ptr<libcamera::Request> &request : requests) {
    request_parked.at(request.get()) = false;
    if (const int ret = camera->queueRequest(request.get()); ret < 0) {
      RCLCPP_ERROR_STREAM(get_logger(), "watchdog: failed to queue request (" << request->toString() << "): " << strerror(-ret));
      return false;
    }
    request_in_camera.at(request.get()) = true;
  }
  RCLCPP_INFO_STREAM(get_logger(), "watchdog: camera restarted, " << requests.size() << " requests queued");
  return true;
}

void
CameraNode::postParameterChange(const std::vector<rclcpp::Parameter> &parameters)
{
  // check non-control parameters
  for (const rclcpp::Parameter &parameter : parameters) {
    if (parameter.get_name() == "jpeg_quality") {
      jpeg_quality = parameter.get_parameter_value().get<uint8_t>();
    }
  }
}

#ifndef RCLCPP_HAS_PARAM_EXT_CB
rcl_interfaces::msg::SetParametersResult
CameraNode::onParameterChange(const std::vector<rclcpp::Parameter> &parameters)
{
  postParameterChange(parameters);

  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  return result;
}
#endif

} // namespace camera
