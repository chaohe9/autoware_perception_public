// Copyright 2020 Tier IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//
#define EIGEN_MPL2_ONLY

#include "multi_object_tracker_node.hpp"

#include "autoware/multi_object_tracker/utils/utils.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <boost/optional.hpp>

#include <glog/logging.h>
#include <tf2_ros/create_timer_interface.h>
#include <tf2_ros/create_timer_ros.h>

#include <iterator>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{
// Function to get the transform between two frames
boost::optional<geometry_msgs::msg::Transform> getTransformAnonymous(
  const tf2_ros::Buffer & tf_buffer, const std::string & source_frame_id,
  const std::string & target_frame_id, const rclcpp::Time & time)
{
  try {
    // Check if the frames are ready
    std::string errstr;  // This argument prevents error msg from being displayed in the terminal.
    if (!tf_buffer.canTransform(
          target_frame_id, source_frame_id, tf2::TimePointZero, tf2::Duration::zero(), &errstr)) {
      return boost::none;
    }

    // Lookup the transform
    geometry_msgs::msg::TransformStamped self_transform_stamped;
    self_transform_stamped = tf_buffer.lookupTransform(
      /*target*/ target_frame_id, /*src*/ source_frame_id, time,
      rclcpp::Duration::from_seconds(0.5));
    return self_transform_stamped.transform;
  } catch (tf2::TransformException & ex) {
    RCLCPP_WARN_STREAM(rclcpp::get_logger("multi_object_tracker"), ex.what());
    return boost::none;
  }
}

}  // namespace

namespace autoware::multi_object_tracker
{
using Label = autoware_perception_msgs::msg::ObjectClassification;

MultiObjectTracker::MultiObjectTracker(const rclcpp::NodeOptions & node_options)
: rclcpp::Node("multi_object_tracker", node_options),
  tf_buffer_(this->get_clock()),
  tf_listener_(tf_buffer_),
  last_published_time_(this->now())
{
  // glog for debug
  if (!google::IsGoogleLoggingInitialized()) {
    google::InitGoogleLogging("multi_object_tracker");
    google::InstallFailureSignalHandler();
  }

  // Get parameters
  double publish_rate = declare_parameter<double>("publish_rate");  // [hz]
  world_frame_id_ = declare_parameter<std::string>("world_frame_id");
  bool enable_delay_compensation{declare_parameter<bool>("enable_delay_compensation")};

  declare_parameter("selected_input_channels", std::vector<std::string>());
  std::vector<std::string> selected_input_channels =
    get_parameter("selected_input_channels").as_string_array();

  // ROS interface - Publisher
  tracked_objects_pub_ = create_publisher<TrackedObjects>("output", rclcpp::QoS{1});

  // ROS interface - Input channels
  // Get input channels
  std::vector<std::string> input_topic_names;
  std::vector<std::string> input_names_long;
  std::vector<std::string> input_names_short;
  std::vector<bool> input_is_spawn_enabled;

  if (selected_input_channels.empty()) {
    RCLCPP_ERROR(this->get_logger(), "No input topics are specified.");
    return;
  }
  else {
    // Selected input channels: 1
    //   - lidar_centerpoint_validated
    RCLCPP_INFO(this->get_logger(), "Selected input channels: %ld", selected_input_channels.size());
    for (const auto & selected_input_channel : selected_input_channels) {
      RCLCPP_INFO(this->get_logger(), "  - %s", selected_input_channel.c_str());
    }
  }

  for (const auto & selected_input_channel : selected_input_channels) {
    // required parameters, no default value
    const std::string input_topic_name =
      declare_parameter<std::string>("input_channels." + selected_input_channel + ".topic");
    input_topic_names.push_back(input_topic_name);

    // required parameter, but can set a default value
    const bool spawn_enabled = declare_parameter<bool>(
      "input_channels." + selected_input_channel + ".can_spawn_new_tracker", true);
    input_is_spawn_enabled.push_back(spawn_enabled);

    // optional parameters
    const std::string default_name = selected_input_channel;
    const std::string name_long = declare_parameter<std::string>(
      "input_channels." + selected_input_channel + ".optional.name", default_name);
    input_names_long.push_back(name_long);

    const std::string default_name_short = selected_input_channel.substr(0, 3);
    const std::string name_short = declare_parameter<std::string>(
      "input_channels." + selected_input_channel + ".optional.short_name", default_name_short);
    input_names_short.push_back(name_short);
  }

  input_channel_size_ = input_topic_names.size();
  input_channels_.resize(input_channel_size_);

  for (size_t i = 0; i < input_channel_size_; i++) {
    input_channels_[i].input_topic = input_topic_names[i];
    input_channels_[i].long_name = input_names_long[i];
    input_channels_[i].short_name = input_names_short[i];
    input_channels_[i].is_spawn_enabled = input_is_spawn_enabled[i];
  }

  // Initialize input manager
  input_manager_ = std::make_unique<InputManager>(*this);
  input_manager_->init(input_channels_);  // Initialize input manager, set subscriptions
  input_manager_->setTriggerFunction(
    std::bind(&MultiObjectTracker::onTrigger, this));  // Set trigger function

  // Create tf timer
  auto cti = std::make_shared<tf2_ros::CreateTimerROS>(
    this->get_node_base_interface(), this->get_node_timers_interface());
  tf_buffer_.setCreateTimerInterface(cti);

  // Create ROS time based timer.
  // If the delay compensation is enabled, the timer is used to publish the output at the correct
  // time.
  if (enable_delay_compensation) {
    const auto timer_period = rclcpp::Rate(publish_rate).period();
    publish_timer_ = rclcpp::create_timer(
      this, get_clock(), timer_period, std::bind(&MultiObjectTracker::onTimer, this));
  }

  // Initialize processor
  {
    std::map<std::uint8_t, std::string> tracker_map;
    tracker_map.insert(
      std::make_pair(Label::CAR, this->declare_parameter<std::string>("car_tracker")));
    tracker_map.insert(
      std::make_pair(Label::TRUCK, this->declare_parameter<std::string>("truck_tracker")));
    tracker_map.insert(
      std::make_pair(Label::BUS, this->declare_parameter<std::string>("bus_tracker")));
    tracker_map.insert(
      std::make_pair(Label::TRAILER, this->declare_parameter<std::string>("trailer_tracker")));
    tracker_map.insert(std::make_pair(
      Label::PEDESTRIAN, this->declare_parameter<std::string>("pedestrian_tracker")));
    tracker_map.insert(
      std::make_pair(Label::BICYCLE, this->declare_parameter<std::string>("bicycle_tracker")));
    tracker_map.insert(std::make_pair(
      Label::MOTORCYCLE, this->declare_parameter<std::string>("motorcycle_tracker")));

    processor_ = std::make_unique<TrackerProcessor>(tracker_map, input_channel_size_);
  }

  // Data association initialization
  {
    const auto tmp = this->declare_parameter<std::vector<int64_t>>("can_assign_matrix");
    const std::vector<int> can_assign_matrix(tmp.begin(), tmp.end());
    const auto max_dist_matrix = this->declare_parameter<std::vector<double>>("max_dist_matrix");
    const auto max_area_matrix = this->declare_parameter<std::vector<double>>("max_area_matrix");
    const auto min_area_matrix = this->declare_parameter<std::vector<double>>("min_area_matrix");
    const auto max_rad_matrix = this->declare_parameter<std::vector<double>>("max_rad_matrix");
    const auto min_iou_matrix = this->declare_parameter<std::vector<double>>("min_iou_matrix");

    association_ = std::make_unique<DataAssociation>(
      can_assign_matrix, max_dist_matrix, max_area_matrix, min_area_matrix, max_rad_matrix,
      min_iou_matrix);
  }

  // Debugger
  debugger_ = std::make_unique<TrackerDebugger>(*this, world_frame_id_);
  debugger_->setObjectChannels(input_names_short);
  published_time_publisher_ =
    std::make_unique<autoware::universe_utils::PublishedTimePublisher>(this);
}

void MultiObjectTracker::onTrigger()
{
  const rclcpp::Time current_time = this->now();
  // get objects from the input manager and run process
  ObjectsList objects_list;
  const bool is_objects_ready = input_manager_->getObjects(current_time, objects_list);
  if (!is_objects_ready) return;
  onMessage(objects_list);

  // Publish without delay compensation
  if (!publish_timer_) {
    const auto latest_object_time = rclcpp::Time(objects_list.back().second.header.stamp);
    checkAndPublish(latest_object_time);
  }
}

void MultiObjectTracker::onTimer()
{
  const rclcpp::Time current_time = this->now();

  // get objects from the input manager and run process
  ObjectsList objects_list;
  const bool is_objects_ready = input_manager_->getObjects(current_time, objects_list);
  if (is_objects_ready) {
    onMessage(objects_list);
  }

  // Publish with delay compensation
  checkAndPublish(current_time);
}

void MultiObjectTracker::onMessage(const ObjectsList & objects_list)
{
  auto tracking_start = this->now().seconds();
  const rclcpp::Time current_time = this->now();
  const rclcpp::Time oldest_time(objects_list.front().second.header.stamp);

  // process start
  debugger_->startMeasurementTime(this->now(), oldest_time);


  // ===== 打印输入目标信息 =====
  size_t total_objects = 0;
  RCLCPP_INFO(this->get_logger(), " ");
  RCLCPP_INFO(this->get_logger(), "=== Input Objects ===");
  RCLCPP_INFO(this->get_logger(), "Current time: %.6f", current_time.seconds());
  
  for (const auto & objects_data : objects_list) {
    const auto & detected_objects = objects_data.second;

    // 获取消息头的时间戳
    rclcpp::Time msg_time(detected_objects.header.stamp);
    double msg_age = (current_time - msg_time).seconds() * 1000.0;  // 毫秒
    
    RCLCPP_INFO(this->get_logger(), 
        "Channel #%u: %zu objects, msg_time=%.6f, age=%.3f ms",
        objects_data.first,
        detected_objects.objects.size(),
        msg_time.seconds(),
        msg_age);
    
    for (size_t i = 0; i < detected_objects.objects.size(); ++i) {
      const auto & obj = detected_objects.objects[i];
      
      double pos_x = obj.kinematics.pose_with_covariance.pose.position.x;
      double pos_y = obj.kinematics.pose_with_covariance.pose.position.y;
      
      double vel_x = 0.0, vel_y = 0.0;
      if (obj.kinematics.has_twist) {
        vel_x = obj.kinematics.twist_with_covariance.twist.linear.x;
        vel_y = obj.kinematics.twist_with_covariance.twist.linear.y;
      }
      double velocity = std::sqrt(vel_x * vel_x + vel_y * vel_y);
      
      RCLCPP_INFO(this->get_logger(),
          "  [%zu] Pos: (%.3f, %.3f) | Vel: (%.3f, %.3f) | Speed: %.3f m/s (%.2f km/h)",
          i, pos_x, pos_y, vel_x, vel_y, velocity, velocity * 3.6);
      
      total_objects++;
    }
  }
  
  // size_t num_trackers_before = processor_->getListTracker().size();

  // run process for each DetectedObjects
  for (const auto & objects_data : objects_list) {
    runProcess(objects_data.second, objects_data.first);
  }
  // process end
  debugger_->endMeasurementTime(this->now());

    // 时间戳2：结束
  auto tracking_end = this->now().seconds();
  double tracking_duration = tracking_end - tracking_start;
  
  // size_t num_trackers_after = processor_->getListTracker().size();
  RCLCPP_INFO(this->get_logger(), "=== Output Trackers ===");
  
  const auto & list_tracker = processor_->getListTracker();
  size_t tracker_index = 0;
  
  for (const auto & tracker : list_tracker) {
    autoware_perception_msgs::msg::TrackedObject tracked_object;
    
    // 使用 getTrackedObject 方法
    if (tracker->getTrackedObject(current_time, tracked_object)) {
      double pos_x = tracked_object.kinematics.pose_with_covariance.pose.position.x;
      double pos_y = tracked_object.kinematics.pose_with_covariance.pose.position.y;
      
      double vel_x = 0.0, vel_y = 0.0;
      // if (tracked_object.kinematics.has_twist) {
        vel_x = tracked_object.kinematics.twist_with_covariance.twist.linear.x;
        vel_y = tracked_object.kinematics.twist_with_covariance.twist.linear.y;
      // }
      double velocity = std::sqrt(vel_x * vel_x + vel_y * vel_y);
      
      RCLCPP_INFO(this->get_logger(),
          "  [%zu] Pos: (%.3f, %.3f) | Vel: (%.3f, %.3f) | Speed: %.3f m/s (%.2f km/h)",
          tracker_index, pos_x, pos_y, vel_x, vel_y, velocity, velocity * 3.6);
    }
    
    tracker_index++;
  }

  // 输出统计
  // static int call_count = 0;
  // call_count++;
  
  // RCLCPP_INFO(this->get_logger(),
  //     "onMessage call #%d", call_count);
  
  RCLCPP_INFO(this->get_logger(),
      "Msg time: %.6f, Now: %.6f, Diff: %.3f ms, Objects: %zu",
      oldest_time.seconds(), current_time.seconds(),
      (current_time - oldest_time).seconds() * 1000.0, total_objects);
  
  //
  RCLCPP_WARN(this->get_logger(),
      "Tracking: %.6f s for %zu objects (%.6f s/object)",
      tracking_duration,
      total_objects,
      total_objects > 0 ? tracking_duration / total_objects : 0.0);
}

void MultiObjectTracker::runProcess(
  const DetectedObjects & input_objects_msg, const uint & channel_index)
{
  // Get the time of the measurement
  const rclcpp::Time measurement_time =
    rclcpp::Time(input_objects_msg.header.stamp, this->now().get_clock_type());

  // Get the transform of the self frame
  const auto self_transform =
    getTransformAnonymous(tf_buffer_, "base_link", world_frame_id_, measurement_time);
  if (!self_transform) {
    RCLCPP_WARN(this->get_logger(), "Failed to get self transform");
    return;
  }

  // Transform the objects to the world frame
  DetectedObjects transformed_objects;
  if (!object_recognition_utils::transformObjects(
        input_objects_msg, world_frame_id_, tf_buffer_, transformed_objects)) {
    RCLCPP_WARN(this->get_logger(), "Failed to transformObjects()");
    return;
  }

  /* prediction */
  processor_->predict(measurement_time);

  /* object association */
  std::unordered_map<int, int> direct_assignment, reverse_assignment;
  {
    const auto & list_tracker = processor_->getListTracker();
    const auto & detected_objects = transformed_objects;
    // global nearest neighbor
    Eigen::MatrixXd score_matrix = association_->calcScoreMatrix(
      detected_objects, list_tracker);  // row : tracker, col : measurement
    association_->assign(score_matrix, direct_assignment, reverse_assignment);

    // Collect debug information - tracker list, existence probabilities, association results
    debugger_->collectObjectInfo(
      measurement_time, processor_->getListTracker(), channel_index, transformed_objects,
      direct_assignment, reverse_assignment);
  }

  /* tracker update */
  processor_->update(transformed_objects, *self_transform, direct_assignment, channel_index);

  /* tracker pruning */
  processor_->prune(measurement_time);

  /* spawn new tracker */
  if (input_manager_->isChannelSpawnEnabled(channel_index)) {
    processor_->spawn(transformed_objects, *self_transform, reverse_assignment, channel_index);
  }
}

void MultiObjectTracker::checkAndPublish(const rclcpp::Time & time)
{
  static int checkAndPublish_count = 0;
  checkAndPublish_count++;
  // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "checkAndPublish call #%d", checkAndPublish_count);
  /* tracker pruning*/
  processor_->prune(time);

  // Publish
  publish(time);

  // Update last published time
  last_published_time_ = this->now();
}

void MultiObjectTracker::publish(const rclcpp::Time & time) const
{
  debugger_->startPublishTime(this->now());
  const auto subscriber_count = tracked_objects_pub_->get_subscription_count() +
                                tracked_objects_pub_->get_intra_process_subscription_count();
  if (subscriber_count < 1) {
    return;
  }
  // Create output msg
  TrackedObjects output_msg, tentative_objects_msg;
  output_msg.header.frame_id = world_frame_id_;
  processor_->getTrackedObjects(time, output_msg);

  // Publish
  tracked_objects_pub_->publish(output_msg);
  published_time_publisher_->publish_if_subscribed(tracked_objects_pub_, output_msg.header.stamp);

  // Publish debugger information if enabled
  debugger_->endPublishTime(this->now(), time);

  if (debugger_->shouldPublishTentativeObjects()) {
    TrackedObjects tentative_output_msg;
    tentative_output_msg.header.frame_id = world_frame_id_;
    processor_->getTentativeObjects(time, tentative_output_msg);
    debugger_->publishTentativeObjects(tentative_output_msg);
  }
  debugger_->publishObjectsMarkers();
}

}  // namespace autoware::multi_object_tracker

#include <rclcpp_components/register_node_macro.hpp>

RCLCPP_COMPONENTS_REGISTER_NODE(autoware::multi_object_tracker::MultiObjectTracker)
