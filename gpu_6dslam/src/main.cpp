#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <ros/subscriber.h>
#include <tf/transform_listener.h>
#include "tf_conversions/tf_eigen.h"
#include <std_srvs/Trigger.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <std_srvs/Empty.h>
#include <gpu_6dslam_msgs/regsiterAll.h>
#include "gpu6DSLAM.h"

#include <visualization_msgs/Marker.h>
#include <unistd.h>
#include <stdio.h>
tf::TransformListener* tf_listener;
ros::Subscriber subscriber_pointcloud2;
ros::Publisher publisher_metascan;
ros::Publisher publisher_rawpc_round_robot;
ros::Publisher marker_pub;
ros::Subscriber subscriber_initial_robot_pose;

std::string frame_global;// = "map";
std::string frame_robot;// = "base_link";
std::string frame_map = "/map";
std::string root_folder_name = "/tmp/slam";
std::string unit3D_frame_id = "m3d_test/m3d_link";//"/base";

gpu6DSLAM *slam;//(root_folder_name);
//Eigen::Affine3f robot_pose = Eigen::Affine3f::Identity();


void callbackPointcloud2(const sensor_msgs::PointCloud2::ConstPtr& msg);
std::string tf_resolve(const std::string& prefix, const std::string& frame_name);

//
bool registerAll(gpu_6dslam_msgs::regsiterAll::Request  &req,
         gpu_6dslam_msgs::regsiterAll::Response &res)
{
	try
	{
		std::cout << "registerAll triggered " << std::endl;

		slam->slam_search_radius_register_all = req.slam_search_radius_register_all;

		if(req.slam_search_radius_register_all > 0.5)
		{
			slam->slam_bucket_size_step_register_all = req.slam_search_radius_register_all;
		}else
		{
			slam->slam_bucket_size_step_register_all = 0.5f;
		}

		for(int i = 0 ; i < 5 ; i++)slam->registerAll();

		pcl::PointCloud<lidar_pointcloud::PointXYZIRNLRGB> metascan = slam->getMetascan();
		std::cout << "Size metascan: " << metascan.size() << std::endl;

		pcl::PCLPointCloud2 pcl_pc2;
		pcl::toPCLPointCloud2(metascan,pcl_pc2);
		sensor_msgs::PointCloud2 cloud;
		pcl_conversions::fromPCL(pcl_pc2,cloud);

		cloud.header.frame_id = frame_global;//frame_map;// TODO
		cloud.header.stamp = ros::Time::now();

		publisher_metascan.publish(cloud);
		ROS_INFO("Publish metascan done");

		res.result = "Ok";
	}
	catch (thrust::system_error e)
	{
		 std::cerr << "Error: " << e.what() << std::endl;
	}
  return true;
}

bool publishMetascan(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res)
{
    pcl::PointCloud<lidar_pointcloud::PointXYZIRNLRGB> metascan = slam->getMetascan();
    std::cout << "Size metascan: " << metascan.size() << std::endl;

    pcl::PCLPointCloud2 pcl_pc2;
    pcl::toPCLPointCloud2(metascan,pcl_pc2);
    sensor_msgs::PointCloud2 cloud;
    pcl_conversions::fromPCL(pcl_pc2,cloud);

    cloud.header.frame_id = frame_global;//frame_map;// TODO
    cloud.header.stamp = ros::Time::now();

    publisher_metascan.publish(cloud);
    ROS_INFO("Publish metascan done");
    return true;
}

//service
// slam.slam_search_radius_register_all = 1.0f;
// slam.slam_bucket_size_step_register_all = 1.0f;
// slam.registerAll();

void callbackInitialPose( const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg )
{
	std::cout << "callbackInitialPose ToDo" << std::endl;
	/*tf::Stamped<tf::Pose> tf_pose;
	//tf::StampedTransform transform;
	geometry_msgs::PoseStamped poseStamped;
	poseStamped.header = msg->header;
	poseStamped.pose.orientation = msg->pose.pose.orientation;
	poseStamped.pose.position = msg->pose.pose.position;
	tf::poseStampedMsgToTF( poseStamped, tf_pose );
	//tf_pose.setData(transform * tf_pose);
	//tf_pose.setData(transform);
	//tf_pose.stamp_ = transform.stamp_;
	//tf_pose.frame_id_ = frame_global;

	Eigen::Affine3d transform_eigen_d = Eigen::Affine3d::Identity();
	tf::transformTFToEigen (tf_pose, transform_eigen_d);
	Eigen::Affine3f transform_eigen = transform_eigen_d.cast<float>();

	slam.callbackInitialPose(transform_eigen);



	pcl::PointCloud<lidar_pointcloud::PointXYZIRNLRGB> metascan = slam.getMetascan();
	std::cout << "Size metascan: " << metascan.size() << std::endl;

	pcl::PCLPointCloud2 pcl_pc2;
	pcl::toPCLPointCloud2(metascan,pcl_pc2);
	sensor_msgs::PointCloud2 cloud;
	pcl_conversions::fromPCL(pcl_pc2,cloud);

	cloud.header.frame_id = frame_global;//frame_map;// TODO
	cloud.header.stamp = ros::Time::now();

	publisher_metascan.publish(cloud);
	ROS_INFO("Publish metascan done");*/
	/*try
	{
		if (msg->header.frame_id.empty())
		{
			ROS_WARN("callback_navPose: pose->header.frame_id empty");
			return;
		}

		if (!tf_listener->canTransform( frame_global, msg->header.frame_id, ros::Time(0)))
		{
			ROS_WARN("callbackInitialPose: Transformation from default frame '%s' to my '%s' not possible", msg->header.frame_id.c_str(), frame_map.c_str());
			return;
		}

		tf_listener->waitForTransform(frame_global, msg->header.frame_id, msg->header.stamp, ros::Duration(1.0));

		geometry_msgs::PoseStamped newer_pose;
		tf::Stamped<tf::Pose> tf_pose;
		tf::StampedTransform transform;
		tf_listener->lookupTransform( frame_global, msg->header.frame_id, ros::Time(0), transform );


		if (!tf_listener->canTransform( frame_global, unit3D_frame_id, ros::Time(0)))
		{
			ROS_WARN("callbackInitialPose: Transformation from default frame '%s' to my '%s' not possible ... return", frame_global.c_str(), unit3D_frame_id.c_str());
			return;
		}


		tf::StampedTransform unit3D_tfpose;
		tf_listener->waitForTransform(frame_global, unit3D_frame_id, ros::Time(0),	ros::Duration(1.0));
		tf_listener->lookupTransform(frame_global,  unit3D_frame_id, ros::Time(0), unit3D_tfpose);

		///////////////////
		geometry_msgs::PoseStamped poseStamped;
		poseStamped.header = msg->header;
		poseStamped.pose.orientation = msg->pose.pose.orientation;
		poseStamped.pose.position = msg->pose.pose.position;
		tf::poseStampedMsgToTF( poseStamped, tf_pose );
		tf_pose.setData(transform * tf_pose);
		tf_pose.stamp_ = transform.stamp_;
		tf_pose.frame_id_ = frame_global;

		Eigen::Affine3d transform_eigen_d = Eigen::Affine3d::Identity();
		tf::transformTFToEigen (tf_pose, transform_eigen_d);
		Eigen::Affine3f transform_eigen = transform_eigen_d.cast<float>();

		Eigen::Affine3d dm = Eigen::Affine3d::Identity();
		tf::transformTFToEigen (unit3D_tfpose, dm);
		Eigen::Affine3f unit3D_pose = dm.cast<float>();

		std::cout << "initial pose" << std::endl;
		std::cout << transform_eigen.matrix() << std::endl;

		std::cout << "unit3D pose" << std::endl;
		std::cout << unit3D_pose.matrix() << std::endl;

		slam.callbackInitialPose(transform_eigen, unit3D_pose);

		//////////////
		Eigen::Affine3f m = Eigen::Affine3f::Identity();
		pcl::PointCloud<lidar_pointcloud::PointXYZIRNLRGB> metascan = slam.getMetascan(m);
		std::cout << "Size metascan: " << metascan.size() << std::endl;

		pcl::PCLPointCloud2 pcl_pc2;
		pcl::toPCLPointCloud2(metascan,pcl_pc2);
		sensor_msgs::PointCloud2 cloud;
		pcl_conversions::fromPCL(pcl_pc2,cloud);

		cloud.header.frame_id = frame_global;//frame_map;// TODO
		cloud.header.stamp = ros::Time::now();

		publisher_metascan.publish(cloud);
		ROS_INFO("Publish metascan done");
	}
	catch (tf::TransformException &ex)
	{
		ROS_ERROR("%s", ex.what());
	}
	*/
	return;
}

int main(int argc, char *argv[])
{
	ros::init(argc, argv, "gpu6dslam_node");
	ros::NodeHandle private_node("~");
	ros::NodeHandle public_node("");

	tf_listener = new tf::TransformListener(ros::Duration(60));

	////////////////////////////////////////////////////////////////////////
	ROS_INFO("reading parameters");
	std::string topic_pointcloud2;
	//private_node.param<std::string>("topic_pointcloud2", topic_pointcloud2, "/unit_sync/stopScanOutput");
	//private_node.param<std::string>("topic_pointcloud2", topic_pointcloud2, "/color_pc/output");
	private_node.param<std::string>("topic_pointcloud2", topic_pointcloud2, "/m3d_test/aggregator/cloud");

	ROS_INFO("param topic_pointcloud2: '%s'", topic_pointcloud2.c_str());

	std::string tf_prefix = tf::getPrefixParam(public_node);
	ROS_INFO("param tf_prefix: '%s'", tf_prefix.c_str());

	private_node.param<std::string>("frame_global", frame_global, "odom");
	frame_global = tf_resolve(tf_prefix, frame_global);
	ROS_INFO("param frame_global: '%s'", frame_global.c_str());

	private_node.param<std::string>("frame_robot", frame_robot, "base_link");
	frame_robot = tf_resolve(tf_prefix, frame_robot);
	ROS_INFO("param frame_robot: '%s'", frame_robot.c_str());

	private_node.param<std::string>("root_folder_name", root_folder_name, "/tmp/slam");

	std::stringstream ss;
	ss << time(0);//ros::Time::now().sec;
	//ss.str()

	//slam->registerSingleScan(pc, m, ss.str());

	std::string root_folder_name_lastest = 	root_folder_name+"_last";
	root_folder_name+=ss.str();

	int err = 0;
	err = remove(root_folder_name_lastest.c_str());
	if (err) ROS_FATAL("Cannot execute remove(%s), return : %d", root_folder_name_lastest.c_str(), err);
	symlink((root_folder_name+"/").c_str(), root_folder_name_lastest.c_str());
	if (err) ROS_FATAL("Cannot execute symlink(%s, %s), return : %d", (root_folder_name+"/").c_str(), root_folder_name_lastest.c_str(), err);
	ROS_INFO("param root_folder_name: '%s'", root_folder_name.c_str());

	//gpu6DSLAM *;//(root_folder_name);


	slam = new gpu6DSLAM(root_folder_name);

    private_node.param<float>("noise_removal_resolution", slam->noise_removal_resolution,  0.5f);
    private_node.param<int>("noise_removal_number_of_points_in_bucket_threshold", slam->noise_removal_number_of_points_in_bucket_threshold,  3);
    private_node.param<float>("noise_removal_bounding_box_extension", slam->noise_removal_bounding_box_extension, 1.0f);

    private_node.param<float>("downsampling_resolution", slam->downsampling_resolution,  0.3f);


    private_node.param<float>("semantic_classification_normal_vectors_search_radius", slam->semantic_classification_normal_vectors_search_radius,  1.0f);
    private_node.param<float>("semantic_classification_curvature_threshold", slam->semantic_classification_curvature_threshold, 10.0);
    private_node.param<float>("semantic_classification_ground_Z_coordinate_threshold", slam->semantic_classification_ground_Z_coordinate_threshold,  1.0f);
    private_node.param<int>("semantic_classification_number_of_points_needed_for_plane_threshold", slam->semantic_classification_number_of_points_needed_for_plane_threshold,  15);
    private_node.param<int>("semantic_classification_max_number_considered_in_INNER_bucket", slam->semantic_classification_max_number_considered_in_INNER_bucket, 100);
    private_node.param<int>("semantic_classification_max_number_considered_in_OUTER_bucket", slam->semantic_classification_max_number_considered_in_OUTER_bucket,  100);
    private_node.param<float>("semantic_classification_bounding_box_extension", slam->semantic_classification_bounding_box_extension, 1.0f);

    private_node.param<float>("slam_registerLastArrivedScan_distance_threshold", slam->slam_registerLastArrivedScan_distance_threshold, 100.0f);
    private_node.param<float>("slam_registerAll_distance_threshold", slam->slam_registerAll_distance_threshold,  10.0f);

    int slam_number_of_observations_threshold;
    private_node.param<int>("slam_number_of_observations_threshold", slam_number_of_observations_threshold,  100);
    slam->slam_number_of_observations_threshold = static_cast<size_t> (slam_number_of_observations_threshold);

    private_node.param<float>("slam_search_radius_step1", slam->slam_search_radius_step1, 2.5f);
    private_node.param<float>("slam_bucket_size_step1", slam->slam_bucket_size_step1,  2.5f);
    private_node.param<int>("slam_registerLastArrivedScan_number_of_iterations_step1", slam->slam_registerLastArrivedScan_number_of_iterations_step1, 30);

    private_node.param<float>("slam_search_radius_step2", slam->slam_search_radius_step2, 2.0f);
    private_node.param<float>("slam_bucket_size_step2", slam->slam_bucket_size_step2,  2.0f);
    private_node.param<int>("slam_registerLastArrivedScan_number_of_iterations_step2", slam->slam_registerLastArrivedScan_number_of_iterations_step2,  30);

    private_node.param<float>("slam_search_radius_step3", slam->slam_search_radius_step3, 1.0f);
    private_node.param<float>("slam_bucket_size_step3", slam->slam_bucket_size_step3, 1.0f);
    private_node.param<int>("slam_registerLastArrivedScan_number_of_iterations_step3", slam->slam_registerLastArrivedScan_number_of_iterations_step3, 30);

    private_node.param<int>("slam_registerAll_number_of_iterations_step1", slam->slam_registerAll_number_of_iterations_step1, 10);
    private_node.param<int>("slam_registerAll_number_of_iterations_step2", slam->slam_registerAll_number_of_iterations_step2, 10);
    private_node.param<int>("slam_registerAll_number_of_iterations_step3", slam->slam_registerAll_number_of_iterations_step3, 10);

    private_node.param<float>("slam_search_radius_register_all", slam->slam_search_radius_register_all, 0.5f);
    private_node.param<float>("slam_bucket_size_step_register_all", slam->slam_bucket_size_step_register_all, 0.5f);

    private_node.param<float>("slam_bounding_box_extension", slam->slam_bounding_box_extension, 1.0f);
    private_node.param<float>("slam_max_number_considered_in_INNER_bucket", slam->slam_max_number_considered_in_INNER_bucket,  100.0f);
    private_node.param<float>("slam_max_number_considered_in_OUTER_bucket", slam->slam_max_number_considered_in_OUTER_bucket,  100.0f);

    private_node.param<float>("slam_observation_weight_plane", slam->slam_observation_weight_plane, 10.0f);
    private_node.param<float>("slam_observation_weight_edge", slam->slam_observation_weight_edge, 1.0f);
    private_node.param<float>("slam_observation_weight_ceiling", slam->slam_observation_weight_ceiling,  10.0f);
    private_node.param<float>("slam_observation_weight_ground", slam->slam_observation_weight_ground,  10.0f);

    private_node.param<float>("findBestYaw_start_angle", slam->findBestYaw_start_angle, -30.0f);
    private_node.param<float>("findBestYaw_finish_angle", slam->findBestYaw_finish_angle,  30.0f);
    private_node.param<float>("findBestYaw_step_angle", slam->findBestYaw_step_angle,  0.5f);
    private_node.param<float>("findBestYaw_bucket_size", slam->findBestYaw_bucket_size,  1.0f);
    private_node.param<float>("findBestYaw_bounding_box_extension", slam->findBestYaw_bounding_box_extension, 1.0f);
    private_node.param<float>("findBestYaw_search_radius", slam->findBestYaw_search_radius,  0.3f);
    private_node.param<float>("findBestYaw_max_number_considered_in_INNER_bucket", slam->findBestYaw_max_number_considered_in_INNER_bucket, 50.0f);
    private_node.param<float>("findBestYaw_max_number_considered_in_OUTER_bucket", slam->findBestYaw_max_number_considered_in_OUTER_bucket,  50.0f);

    private_node.param<bool>("use4DOF",slam->use4DOF,  true);
    private_node.param<int>("cudaDevice",slam->cudaDevice,  0);



    ROS_INFO_STREAM("noise_removal_resolution : " << slam->noise_removal_resolution );
    ROS_INFO_STREAM("noise_removal_number_of_points_in_bucket_threshold : " << slam->noise_removal_number_of_points_in_bucket_threshold );
    ROS_INFO_STREAM("noise_removal_bounding_box_extension : " << slam->noise_removal_bounding_box_extension );
    ROS_INFO_STREAM("downsampling_resolution : " << slam->downsampling_resolution );
    ROS_INFO_STREAM("semantic_classification_normal_vectors_search_radius : " << slam->semantic_classification_normal_vectors_search_radius );
    ROS_INFO_STREAM("semantic_classification_curvature_threshold : " << slam->semantic_classification_curvature_threshold );
    ROS_INFO_STREAM("semantic_classification_ground_Z_coordinate_threshold : " << slam->semantic_classification_ground_Z_coordinate_threshold );
    ROS_INFO_STREAM("semantic_classification_number_of_points_needed_for_plane_threshold : " << slam->semantic_classification_number_of_points_needed_for_plane_threshold );
    ROS_INFO_STREAM("semantic_classification_max_number_considered_in_INNER_bucket : " << slam->semantic_classification_max_number_considered_in_INNER_bucket );
    ROS_INFO_STREAM("semantic_classification_max_number_considered_in_OUTER_bucket : " << slam->semantic_classification_max_number_considered_in_OUTER_bucket );
    ROS_INFO_STREAM("semantic_classification_bounding_box_extension : " << slam->semantic_classification_bounding_box_extension );
    ROS_INFO_STREAM("slam_registerLastArrivedScan_distance_threshold : " << slam->slam_registerLastArrivedScan_distance_threshold );
    ROS_INFO_STREAM("slam_registerAll_distance_threshold : " << slam->slam_registerAll_distance_threshold );
    ROS_INFO_STREAM("slam_number_of_observations_threshold : " << slam->slam_number_of_observations_threshold );
    ROS_INFO_STREAM("slam_search_radius_step1 : " << slam->slam_search_radius_step1 );
    ROS_INFO_STREAM("slam_bucket_size_step1 : " << slam->slam_bucket_size_step1 );
    ROS_INFO_STREAM("slam_registerLastArrivedScan_number_of_iterations_step1 : " << slam->slam_registerLastArrivedScan_number_of_iterations_step1 );
    ROS_INFO_STREAM("slam_search_radius_step2 : " << slam->slam_search_radius_step2 );
    ROS_INFO_STREAM("slam_bucket_size_step2 : " << slam->slam_bucket_size_step2 );
    ROS_INFO_STREAM("slam_registerLastArrivedScan_number_of_iterations_step2 : " << slam->slam_registerLastArrivedScan_number_of_iterations_step2 );
    ROS_INFO_STREAM("slam_search_radius_step3 : " << slam->slam_search_radius_step3 );
    ROS_INFO_STREAM("slam_bucket_size_step3 : " << slam->slam_bucket_size_step3 );
    ROS_INFO_STREAM("slam_registerLastArrivedScan_number_of_iterations_step3 : " << slam->slam_registerLastArrivedScan_number_of_iterations_step3 );
    ROS_INFO_STREAM("slam_registerAll_number_of_iterations_step1 : " << slam->slam_registerAll_number_of_iterations_step1 );
    ROS_INFO_STREAM("slam_registerAll_number_of_iterations_step2 : " << slam->slam_registerAll_number_of_iterations_step2 );
    ROS_INFO_STREAM("slam_registerAll_number_of_iterations_step3 : " << slam->slam_registerAll_number_of_iterations_step3 );
    ROS_INFO_STREAM("slam_search_radius_register_all : " << slam->slam_search_radius_register_all );
    ROS_INFO_STREAM("slam_bucket_size_step_register_all : " << slam->slam_bucket_size_step_register_all );
    ROS_INFO_STREAM("slam_bounding_box_extension : " << slam->slam_bounding_box_extension );
    ROS_INFO_STREAM("slam_max_number_considered_in_INNER_bucket : " << slam->slam_max_number_considered_in_INNER_bucket );
    ROS_INFO_STREAM("slam_max_number_considered_in_OUTER_bucket : " << slam->slam_max_number_considered_in_OUTER_bucket );
    ROS_INFO_STREAM("slam_observation_weight_plane : " << slam->slam_observation_weight_plane );
    ROS_INFO_STREAM("slam_observation_weight_edge : " << slam->slam_observation_weight_edge );
    ROS_INFO_STREAM("slam_observation_weight_ceiling : " << slam->slam_observation_weight_ceiling );
    ROS_INFO_STREAM("slam_observation_weight_ground : " << slam->slam_observation_weight_ground );
    ROS_INFO_STREAM("findBestYaw_start_angle : " << slam->findBestYaw_start_angle );
    ROS_INFO_STREAM("findBestYaw_finish_angle : " << slam->findBestYaw_finish_angle );
    ROS_INFO_STREAM("findBestYaw_step_angle : " << slam->findBestYaw_step_angle );
    ROS_INFO_STREAM("findBestYaw_bucket_size : " << slam->findBestYaw_bucket_size );
    ROS_INFO_STREAM("findBestYaw_bounding_box_extension : " << slam->findBestYaw_bounding_box_extension );
    ROS_INFO_STREAM("findBestYaw_search_radius : " << slam->findBestYaw_search_radius );
    ROS_INFO_STREAM("findBestYaw_max_number_considered_in_INNER_bucket : " << slam->findBestYaw_max_number_considered_in_INNER_bucket );
    ROS_INFO_STREAM("findBestYaw_max_number_considered_in_OUTER_bucket : " << slam->findBestYaw_max_number_considered_in_OUTER_bucket );
    ROS_INFO_STREAM("use4DOF : " << slam->use4DOF );
    ROS_INFO_STREAM("cudaDevice : " << slam->cudaDevice );




	ROS_INFO("setting up topic communication");
	subscriber_pointcloud2 = public_node.subscribe(topic_pointcloud2, 1,
			callbackPointcloud2);
    ros::ServiceServer service1 = public_node.advertiseService("registerAll", registerAll);
    ros::ServiceServer service2 = public_node.advertiseService("publishMetascan", publishMetascan);

	publisher_metascan = public_node.advertise<sensor_msgs::PointCloud2>("/metascan", 1);

	publisher_rawpc_round_robot = public_node.advertise<sensor_msgs::PointCloud2>("/rawpcroundrobot", 1);

	marker_pub = public_node.advertise<visualization_msgs::Marker>("slam_progress_marker", 1);

	subscriber_initial_robot_pose = public_node.subscribe( "initialpose", 1000, &callbackInitialPose );

	///////////////////////////////////////////////////////////////////////////
	int is_slam_loading_map_from_file = 0;
	private_node.param<int>("is_slam_loading_map_from_file", is_slam_loading_map_from_file,  0);
	//slam.slam_number_of_observations_threshold = static_cast<size_t> (slam_number_of_observations_threshold);

	if(is_slam_loading_map_from_file == 1)
	{
		std::string map_filename;
		private_node.param<std::string>("map_filename", map_filename, "/tmp/slam");
		ROS_INFO("param map_filename: '%s'", map_filename.c_str());
		ROS_INFO("trying loading map from file...");

		if(slam->loadmapfromfile(map_filename))
		{
			pcl::PointCloud<lidar_pointcloud::PointXYZIRNLRGB> metascan = slam->getMetascan(Eigen::Affine3f::Identity());
			std::cout << "Size metascan: " << metascan.size() << std::endl;

			pcl::PCLPointCloud2 pcl_pc2;
			pcl::toPCLPointCloud2(metascan,pcl_pc2);
			sensor_msgs::PointCloud2 cloud;
			pcl_conversions::fromPCL(pcl_pc2,cloud);

			cloud.header.frame_id = frame_global;//frame_map;// TODO
			cloud.header.stamp = ros::Time::now();

			publisher_metascan.publish(cloud);
			ROS_INFO("Publish metascan done");
		}
	}



	///////////////////////////////////////////////////////////////////////////


	while (ros::ok())
	//&& (!tf_listener->waitForTransform(frame_global, frame_robot,
	//					ros::Time(), ros::Duration(1.0))))
	{
		ros::spinOnce ();
	}

	std::cout << "!ros::ok() return 0" << std::endl;
	delete slam;
	return 0;
}

void callbackPointcloud2(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
	std::cout << "callbackPointcloud2" << std::endl;

	tf::StampedTransform position_current;
	try
	{
		tf_listener->waitForTransform(frame_global, msg->header.frame_id, msg->header.stamp,
				ros::Duration(1.0));

		//tf_listener->lookupTransform(frame_global, frame_robot, msg->header.stamp,
		//		position_current);

		tf_listener->lookupTransform(frame_global, /*frame_robot*/ msg->header.frame_id, msg->header.stamp, position_current);


        Eigen::Affine3d dm = Eigen::Affine3d::Identity();

        tf::transformTFToEigen (position_current, dm);

        Eigen::Affine3f m = dm.cast<float>();

       // robot_pose = m;

		pcl::PointCloud<lidar_pointcloud::PointXYZIRNLRGB> pc;
		pcl::fromROSMsg(*msg, pc);

		if(pc.points.size() < 1000){
			std::cout << "pc.points.size() < 1000 ---INPUT CLOUD TO SMALL!!!--- return" << std::endl;
			return;
		}

		try
		{
			// std::stringstream ss;
      // ss << msg->header.stamp.sec;
      char txt[10];
      sprintf(txt,"%08d", msg->header.seq);
		    ////////////////

		   /* visualization_msgs::Marker text;
		    text.header.frame_id = frame_robot;
		    text.header.stamp = ros::Time::now();
		    //text.ns = "lines";
		    text.action = visualization_msgs::Marker::MODIFY;


		    text.id = 2001;
		    text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;

		    text.pose.position.z = 3.0;
		    text.pose.orientation.w = 1.0;

		    text.scale.x = 1.1;

		    text.color.r = 1.0;
		    text.color.a = 1.0;
		    text.text = "slam START";

		    marker_pub.publish(text);

		    /////////////////
		    ros::spinOnce();*/
      // slam->registerSingleScan(pc, m, ss.str());
			slam->registerSingleScan(pc, m, txt);


			/////////////////////
			//visualization_msgs::Marker text;
			/*text.header.frame_id = frame_robot;
			text.header.stamp = ros::Time::now();
			//text.ns = "lines";
			text.action = visualization_msgs::Marker::MODIFY;


			text.id = 2002;
			text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;

			text.pose.position.z = 3.0;
			text.pose.orientation.w = 1.0;

			text.scale.x = 1.1;

			text.color.r = 1.0;
			text.color.a = 1.0;
			text.text = "slam FINISHED";

			marker_pub.publish(text);
			ros::spinOnce();*/
			////////////////////
		}catch (thrust::system_error e)
		{
			 std::cerr << "Error: " << e.what() << std::endl;
		}

		if(publisher_metascan.getNumSubscribers() > 0)
		{
			pcl::PointCloud<lidar_pointcloud::PointXYZIRNLRGB> metascan = slam->getMetascan(m);
			std::cout << "Size metascan: " << metascan.size() << std::endl;

			pcl::PCLPointCloud2 pcl_pc2;
			pcl::toPCLPointCloud2(metascan,pcl_pc2);
			sensor_msgs::PointCloud2 cloud;
			pcl_conversions::fromPCL(pcl_pc2,cloud);

			cloud.header.frame_id = frame_global;//frame_map;// TODO
			cloud.header.stamp = ros::Time::now();

			publisher_metascan.publish(cloud);
			ROS_INFO("Publish metascan done");
		}

		////////////////////////////
		if(publisher_rawpc_round_robot.getNumSubscribers() > 0)
		{
			pcl::PointCloud<lidar_pointcloud::PointXYZIRNLRGB> _pc;
			for(size_t i = 0 ; i < pc.points.size(); i++)
			{
				lidar_pointcloud::PointXYZIRNLRGB p = pc[i];

				float dist = sqrtf ( p.x * p.x + p.y * p.y + p.z * p.z );

				if(dist > 0.5f && dist < 5.0f && p.z < 2.0f)
				{
					_pc.push_back(p);
				}
			}

			slam->downsample(_pc, 0.1);

			slam->transformPointCloud(_pc, m);

			pcl::PCLPointCloud2 pcl_pc2;
			pcl::toPCLPointCloud2(_pc,pcl_pc2);
			sensor_msgs::PointCloud2 cloud;
			pcl_conversions::fromPCL(pcl_pc2,cloud);

			cloud.header.frame_id = frame_global;//frame_map;// TODO
			cloud.header.stamp = ros::Time::now();

			publisher_rawpc_round_robot.publish(cloud);
			ROS_INFO("Publish rawpcroundrobot done");
		}


		//for(size_t i = 0 ; i < pc.points.size(); i++)
		//{
		//	std::cout << pc[i].x << " " << pc[i].y << " " << pc[i].z << " " << pc[i].intensity << " " << pc[i].ring << " " << pc[i].normal_x << " " << pc[i].normal_y << " " << pc[i].normal_z << " " << pc[i].label << " " << pc[i].rgb << std::endl;
		//}


	}catch (tf::TransformException &ex)
	{
		ROS_ERROR("%s", ex.what());
	}

}

std::string tf_resolve(const std::string& prefix,
		const std::string& frame_name) {
	if (frame_name.size() > 0)
		if (frame_name[0] == '/') {
			return frame_name;
		}
	if (prefix.size() > 0) {
		if (prefix[0] == '/') {
			std::string composite = prefix;
			composite.append("/");
			composite.append(frame_name);
			return composite;

		} else {
			std::string composite;
			composite = "/";
			composite.append(prefix);
			composite.append("/");
			composite.append(frame_name);
			return composite;
		}
	} else {
		std::string composite;
		composite = "/";
		composite.append(frame_name);
		return composite;
	}
}
