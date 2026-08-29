#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/empty.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/battery_state.hpp"
#include "cv_bridge/cv_bridge.h"

#define PI 3.14159265359
#define PI2 PI * 2.0
#define DEG_TO_RAD PI / 180

#define NO_KEY -1

#define KEY_NUM_0 48
#define KEY_NUM_1 49
#define KEY_NUM_2 50
#define KEY_NUM_3 51
#define KEY_NUM_4 52
#define KEY_NUM_5 53
#define KEY_NUM_6 54
#define KEY_NUM_7 55
#define KEY_NUM_8 56
#define KEY_NUM_9 57

#define KEY_UP 82
#define KEY_DOWN 84
#define KEY_LEFT 81
#define KEY_RIGHT 83

#define KEY_ENTER 13
#define KEY_SPACE 32

#define WINDOW "Tello"

using namespace std::chrono_literals;

class TelloControl : public rclcpp::Node
{
	public:
		/**
		 * Store the last key pressed when controlling the drone manually.
		 *
		 * Used to detect changes in key pressed.
		 */
		int last_key = NO_KEY;

		/**
		 * Timer used to control the execution speed of the node.
		 */
		rclcpp::TimerBase::SharedPtr timer;

		/**
		 * Publish drone control velocity.
		 */
		rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_velocity;

		/**
		 * Publish takeoff control.
		 */
		rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr publisher_takeoff;

		/**
		 * Publisher for drone flip commands.
		 */
		rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_flip;

		/**
		 * Publisher for landing controls.
		 */
		rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr publisher_land;

		/**
		 * Publisher for emergency stop.
		 */
		rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr publisher_emergency;

		/**
		 * Subscribe to the drone camera image.
		 */
		rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_image;

		/**
		 * Subscribe to drone battery status.
		 */
		rclcpp::Subscription<sensor_msgs::msg::BatteryState>::SharedPtr subscription_battery;

		/**
		 * Last received camera frame.
		 */
		cv::Mat last_frame;

		/**
		 * Whether we have a valid frame to display.
		 */
		bool has_frame = false;

		/**
		 * Last received battery percentage (0..1). -1 means unknown.
		 */
		double battery_percentage = -1.0;

		/**
		 * Construct a new Tello Control object
		 */
		TelloControl() : Node("control")
		{
			publisher_land = this->create_publisher<std_msgs::msg::Empty>("land", 1);
			publisher_flip = this->create_publisher<std_msgs::msg::String>("flip", 1);
			publisher_takeoff = this->create_publisher<std_msgs::msg::Empty>("takeoff", 1);
			publisher_velocity = this->create_publisher<geometry_msgs::msg::Twist>("control", 1);
			publisher_emergency = this->create_publisher<std_msgs::msg::Empty>("emergency", 1);

			subscription_image = this->create_subscription<sensor_msgs::msg::Image>(
				"image_raw",
				rclcpp::SensorDataQoS(),
				std::bind(&TelloControl::imageCallback, this, std::placeholders::_1));

			subscription_battery = this->create_subscription<sensor_msgs::msg::BatteryState>(
				"battery",
				10,
				std::bind(&TelloControl::batteryCallback, this, std::placeholders::_1));

			// One window, created once. Calling namedWindow inside the timer
			// re-creates the window handle every tick on some HighGUI backends.
			cv::namedWindow(WINDOW, cv::WINDOW_AUTOSIZE);

			// 33 ms, not 1 ms. The previous 1 ms timer immediately blocked for
			// 15 ms inside cv::waitKey, so the node actually ran at ~66 Hz while
			// spinning a core to render a 30 Hz video stream twice over. The
			// timer sets the rate; waitKey(1) only pumps the GUI event loop.
			timer = this->create_wall_timer(33ms, std::bind(&TelloControl::timerCallback, this));
		}

		void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
		{
			try
			{
				// Image is published as bgr8 by tello node.
				auto cv_ptr = cv_bridge::toCvCopy(*msg, "bgr8");
				last_frame = cv_ptr->image;
				has_frame = !last_frame.empty();
			}
			catch (const std::exception &e)
			{
				RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
					"Failed to convert image: %s", e.what());
			}
		}

		void batteryCallback(const sensor_msgs::msg::BatteryState::SharedPtr msg)
		{
			battery_percentage = msg->percentage;
		}

		/**
		 * Method to control the drone using the keyboard inputs.
		 *
		 * @param key Keycode received.
		 */
		void manualControl(int key)
		{
			// Stick deflection in SDK units. NOTE the axis convention on the
			// `control` topic is x = lateral, y = forward -- NOT REP-103. It is
			// kept because changing it would invert an operator's controls. The
			// driver also exposes a REP-103-compliant `cmd_vel` topic.
			double manual_speed = 50;

			geometry_msgs::msg::Twist msg = geometry_msgs::msg::Twist();
		
			if(key == KEY_LEFT) {msg.linear.x = -manual_speed;}
			if(key == KEY_RIGHT) {msg.linear.x = +manual_speed;}
			if(key == KEY_UP) {msg.linear.y = manual_speed;}
			if(key == KEY_DOWN) {msg.linear.y = -manual_speed;}
			if(key == (int)('w')) {msg.linear.z = manual_speed;}
			if(key == (int)('s')) {msg.linear.z = -manual_speed;}
			if(key == (int)('a')) {msg.angular.z = -manual_speed;}
			if(key == (int)('d')) {msg.angular.z = manual_speed;}

			publisher_velocity->publish(msg);
		}

		void timerCallback()
		{
			if (!rclcpp::ok())
			{
				return;
			}

			cv::Mat to_show;
			if (has_frame)
			{
				to_show = last_frame.clone();
			}
			else
			{
				to_show = cv::Mat::zeros(240, 320, CV_8UC3);
				cv::putText(to_show, "Waiting for /image_raw ...", cv::Point(10, 120),
					cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
			}

			// Battery overlay
			std::string battery_text = "Battery: N/A";
			if (battery_percentage >= 0.0)
			{
				int pct = static_cast<int>(battery_percentage * 100.0 + 0.5);
				battery_text = "Battery: " + std::to_string(pct) + "%";
			}
			cv::putText(to_show, battery_text, cv::Point(10, 30),
				cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

			// Controls overlay
			cv::putText(to_show, "Controls: T=takeoff  L=land  E=emergency  F=flip(fwd)", cv::Point(10, 55),
				cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1);
			cv::putText(to_show, "Arrows/WASD: move  (W/S up-down, A/D yaw)", cv::Point(10, 75),
				cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1);

			cv::imshow(WINDOW, to_show);

			// waitKey(1) pumps the GUI event queue without blocking the
			// executor; the timer period is what sets the display rate.
			// The mask strips the modifier bits some backends set in the high
			// bytes, which otherwise make every keypress compare unequal.
			int key = cv::waitKey(1);
			if (key != NO_KEY)
			{
				key &= 0xFF;
				// Takeoff
				if(key == (int)('t'))
				{
					publisher_takeoff->publish(std_msgs::msg::Empty());
				}
				// Land
				else if(key == (int)('l'))
				{
					publisher_land->publish(std_msgs::msg::Empty());
				}
				// Flip
				else if(key == (int)('f'))
				{
					std_msgs::msg::String msg = std_msgs::msg::String();
					msg.data = 'f';
					publisher_flip->publish(msg);
				}
				// Emergency Stop
				else if(key == (int)('e'))
				{
					publisher_emergency->publish(std_msgs::msg::Empty());
				}
				else
				{
					manualControl(key);
				}
			}

			// Store last key for diffs
			last_key = key;
		}
};

int main(int argc, char * argv[])
{
	rclcpp::init(argc, argv);
	{
		auto node = std::make_shared<TelloControl>();
		rclcpp::spin(node);
	}
	// Tear the HighGUI window down explicitly. Without this the Qt/GTK backend
	// keeps its event loop alive after spin() returns, so the process ignores
	// SIGINT and launch escalates to SIGTERM and then SIGKILL -- 15 seconds of
	// waiting on every Ctrl-C.
	try {
		cv::destroyAllWindows();
		cv::waitKey(1);
	} catch (const cv::Exception &) {
		// A headless build has no windows to destroy; nothing to do.
	}
	rclcpp::shutdown();
	return 0;
}