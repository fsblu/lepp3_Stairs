#ifndef LEPP2_STAIR_SEGMENTER_H__
#define LEPP2_STAIR_SEGMENTER_H__

#include "lepp2/BaseSegmenter.hpp"

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>

namespace lepp {

/**
 * TODO put comments
 */
template<class PointT>
class StairSegmenter: public BaseSegmenter<PointT> {
public:
	StairSegmenter();

	virtual std::vector<typename pcl::PointCloud<PointT>::ConstPtr> segment(
			const typename pcl::PointCloud<PointT>::ConstPtr& cloud);
private:
	// Helper typedefs to make the implementation code cleaner
	typedef pcl::PointCloud<PointT> PointCloudT;
	typedef typename PointCloudT::Ptr PointCloudPtr;
	typedef typename PointCloudT::ConstPtr CloudConstPtr;

	// Private helper member functions
	/**
	 * Performs some initial preprocessing and filtering appropriate for the
	 * segmentation algorithm.
	 * Takes the original cloud as a parameter and returns a pointer to a newly
	 * created (and allocated) cloud containing the result of the filtering.
	 */
	PointCloudPtr preprocessCloud(CloudConstPtr const& cloud);
	/**
	 * Removes all planes from the given point cloud.
	 */
	void findStairs(PointCloudPtr const& cloud_filtered);
	/**
	 * Extracts the Euclidean clusters from the given point cloud.
	 * Returns a vector where each element represents the pcl::PointIndices
	 * instance representing the corresponding cluster.
	 */
	std::vector<pcl::PointIndices> getStairClusters();
	/**
	 * Convert the clusters represented by the given indices to point clouds,
	 * by copying the corresponding points from the cloud to the corresponding
	 * new point cloud.
	 */
	std::vector<CloudConstPtr> clustersToPointClouds(
			std::vector<pcl::PointIndices> const& cluster_indices);

	/* Classification function between segmentation and clustering steps.
	 * Classify any segmented planar plane into one of the segmented surface group,
	 * regarding its plane normal.
	 * This function is necessary to be able to separate ramps or any inclined
	 * surface from the floor at the clustering step.
	 * */

	bool classify(PointCloudPtr const& cloud_planar_surface,
			const pcl::ModelCoefficients & coeffs);

	/*Returns the angle between two plane normals.
	 * Calculation is made based on plane coefficients
	 */
	double getAngle(const pcl::ModelCoefficients &coeffs, const int &index);

	/*
	 *This function is called if a plane whose plane normal differs more
	 *than 3 degrees with the saved segmented plane groups, is segmented.The
	 *plane and its coefficients are saved under a new plane group.
	 * */

	bool addCoefficient(
			const PointCloudPtr &cloud_planar_surface,
			const pcl::ModelCoefficients &coeffs);

	/*This function is called to add the segmented plane to a corresponding
	 *surface group. Surface planes normals devising 3 degrees are considered same
	 *type of surface and saved together.
	 * */
	bool addPlane(int const index, const PointCloudPtr & cloud_planar_surface);
	/**
	 * Instance used to extract the planes from the input cloud.
	 */
	pcl::SACSegmentation<PointT> segmentation_;
	/**
	 * Instance used to extract the actual clusters from the input cloud.
	 */
	pcl::EuclideanClusterExtraction<PointT> clusterizer_;
	/**
	 * The KdTree will hold the representation of the point cloud which is passed
	 * to the clusterizer.
	 */
	boost::shared_ptr<pcl::search::KdTree<PointT> > kd_tree_;

	/**
	 * The cloud that holds all planar surfaces (Stairs).
	 */
	PointCloudPtr cloud_stairs_;
	/**
	 * The percentage of the original cloud that should be kept for the
	 * clusterization, at the least.
	 * We stop removing planes from the original cloud once there are either no
	 * more planes to be removed or when the number of points remaining in the
	 * cloud dips below this percentage of the original cloud.
	 */

	/**
	 * TEMP: vector containing clouds for each stair
	 */
	std::vector<CloudConstPtr> vec_cloud_stairs_;

	/* The vector containing surface groups, created according to difference in inclination during the segmentation
	 * */
	std::vector<PointCloudPtr> vec_surface;

	/* List of plane coefficients [normal_x normal_y normal_z hessian_component_d]
	 *
	 */
	std::vector<pcl::ModelCoefficients> m_coefficients;

	/*Segmentation ratio*/
	double const min_filter_percentage_;
};

template<class PointT>
StairSegmenter<PointT>::StairSegmenter() :
		min_filter_percentage_(0.1), kd_tree_(
				new pcl::search::KdTree<PointT>()), cloud_stairs_(
				new PointCloudT()) {
	// Parameter initialization of the plane segmentation
	segmentation_.setOptimizeCoefficients(true);
	segmentation_.setModelType(pcl::SACMODEL_PLANE);
	segmentation_.setMethodType(pcl::SAC_RANSAC);
	segmentation_.setMaxIterations(100); // value recognized by Irem
	segmentation_.setDistanceThreshold(0.03);

	// Parameter initialization of the clusterizer
	clusterizer_.setClusterTolerance(0.02);
	clusterizer_.setMinClusterSize(1800);
	clusterizer_.setMaxClusterSize(30000);
}

template<class PointT>
typename pcl::PointCloud<PointT>::Ptr StairSegmenter<PointT>::preprocessCloud(
		CloudConstPtr const& cloud) {
	// Remove NaN points from the input cloud.
	// The pcl API forces us to pass in a reference to the vector, even if we have
	// no use of it later on ourselves.
	PointCloudPtr cloud_filtered(new PointCloudT());
	std::vector<int> index;
	pcl::removeNaNFromPointCloud<PointT>(*cloud, *cloud_filtered, index);

	return cloud_filtered;
}

template<class PointT>
void StairSegmenter<PointT>::findStairs(PointCloudPtr const& cloud_filtered) {

	vec_cloud_stairs_.clear();
	m_coefficients.clear();
	vec_surface.clear();

	// Instance that will be used to perform the elimination of unwanted points
	// from the point cloud.
	pcl::ExtractIndices<PointT> extract;
	// Will hold the indices of the next extracted plane within the loop
	pcl::PointIndices::Ptr current_plane_indices(new pcl::PointIndices);
	// Another instance of when the pcl API requires a parameter that we have no
	// further use for.
	// Remove planes until we reach x % of the original number of points
	size_t const original_cloud_size = cloud_filtered->size();
	size_t const point_threshold = min_filter_percentage_ * original_cloud_size;
	cloud_stairs_->clear();
	while (cloud_filtered->size() > point_threshold) {
		// Try to obtain the next plane...
		pcl::ModelCoefficients coefficients;
		segmentation_.setInputCloud(cloud_filtered);
		segmentation_.segment(*current_plane_indices, coefficients);

		// We didn't get any plane in this run. Therefore, there are no more planes
		// to be removed from the cloud.
		if (current_plane_indices->indices.size() == 0) {
			std::cout << "cannot find more planes > BREAK" << std::endl;
			break;
		}

		// Cloud that holds a plane in each iteration, to be added to the total cloud.
		PointCloudPtr cloud_planar_surface(new PointCloudT());

		// Add the planar inliers to the cloud holding the stairs
		extract.setInputCloud(cloud_filtered);
		extract.setIndices(current_plane_indices);
		extract.setNegative(false);
		extract.filter(*cloud_planar_surface);

		// ... and remove those inliers from the input cloud
		extract.setNegative(true);
		extract.filter(*cloud_filtered);

		*cloud_stairs_ += *cloud_planar_surface;

		//Classify the Cloud
		classify(cloud_planar_surface, coefficients);
		//vec_cloud_stairs_.push_back(cloud_planar_surface);
	}

	std::cout<<"The number of surface groups: "<<vec_surface.size()<<std::endl;
	std::cout<<"The number of coeffs: "<<m_coefficients.size()<<std::endl;

	std::cout << "#found Stairs: " << vec_cloud_stairs_.size() << std::endl;
}

template<class PointT>
std::vector<pcl::PointIndices> StairSegmenter<PointT>::getStairClusters() {
	// Extract the clusters from such a filtered cloud.
	kd_tree_->setInputCloud(cloud_stairs_);
	clusterizer_.setSearchMethod(kd_tree_);
	clusterizer_.setInputCloud(cloud_stairs_);
	std::vector<pcl::PointIndices> cluster_indices;
	clusterizer_.extract(cluster_indices);

	return cluster_indices;
}

template<class PointT> double StairSegmenter<PointT>::getAngle(
		const pcl::ModelCoefficients &coeffs, const int &index) {

	//Scalar Product
	float scalar_product = (m_coefficients.at(index).values[0]
			* coeffs.values[0])
			+ (m_coefficients.at(index).values[1] * coeffs.values[1])
			+ (m_coefficients.at(index).values[2] * coeffs.values[2]);
	double angle = acos(scalar_product) * 180.0 / M_PI;
	cout << "The angle from scalar product: " << angle << std::endl;

	return angle;
}

template<class PointT>
bool StairSegmenter<PointT>::classify(PointCloudPtr const& cloud_planar_surface,
		const pcl::ModelCoefficients & coeffs) {

	bool coeff_exists = false;

	int size = m_coefficients.size();

	for (int i = 0; i < size; i++) {
		double angle=getAngle(coeffs, i);
		cout << "Angle: " << angle << std::endl;

		if (angle < 3|| angle>177) {

			cout << "Expected addition to surface: " << i << std::endl;

			coeff_exists = addPlane(i, cloud_planar_surface);
			break;
		}
	}

	if (coeff_exists == false) {
		coeff_exists = addCoefficient(cloud_planar_surface,
				coeffs);
	}

	if (coeff_exists == false)
		return false;
	return true;
}
template<class PointT>
bool StairSegmenter<PointT>::addCoefficient(
		const PointCloudPtr & cloud_planar_surface,
		const pcl::ModelCoefficients & coeffs) {

	vec_surface.push_back(cloud_planar_surface);
	m_coefficients.push_back(coeffs);

	//bool isPlaneAdded = addClusterParameters(true, 0, size);

	//if (true) {

	cout << "New coefficient and plane added!" << std::endl;
    return true;//}

}
//template<class PointT>
//void Segmentation<PointT>::addClusterParameters(bool isFirstPlane,
//		int surface, int &size) {
//	numberOfSurface.at(surface) += 1;
//	//cout << "beginning of add cluster parameter" << std::endl;
//
//	if (isFirstPlane) {
//		max_points.at(surface) = size;
//		min_points.at(surface) = size;
//
//	} else if (size > max_points[surface])
//		max_points.at(surface) = size;
//	else if (size < min_points[surface])
//		min_points.at(surface) = size;
//	//cout << "end of add cluster parameter" << std::endl;
//
//}
template<class PointT>
bool StairSegmenter<PointT>::addPlane(int const index,
		const PointCloudPtr & cloud_planar_surface) {

	*vec_surface.at(index) += *cloud_planar_surface;
	//addClusterParameters(false, index, size);
	cout << "Added to surface:" << index << std::endl;
	return true;

}
template<class PointT>
std::vector<typename pcl::PointCloud<PointT>::ConstPtr> StairSegmenter<PointT>::clustersToPointClouds(
		std::vector<pcl::PointIndices> const& cluster_indices) {
	// Now copy the points belonging to each cluster to a separate PointCloud
	// and finally return a vector of these point clouds.
	//std::vector<CloudConstPtr> ret;
	size_t const cluster_count = cluster_indices.size();
	for (size_t i = 0; i < cluster_count; ++i) {

		typename PointCloudT::Ptr current(new PointCloudT());
		std::vector<int> const& curr_indices = cluster_indices[i].indices;
		size_t const curr_indices_sz = curr_indices.size();
		for (size_t j = 0; j < curr_indices_sz; j++) {
			// add the point to the corresponding point cloud
			current->push_back(cloud_stairs_->at(curr_indices[j]));
		}

		vec_cloud_stairs_.push_back(current);
	}

	std::cout << "Stairs number:" << vec_cloud_stairs_.size() << std::endl;
	return vec_cloud_stairs_;
}

template<class PointT>
std::vector<typename pcl::PointCloud<PointT>::ConstPtr> StairSegmenter<PointT>::segment(
		const typename pcl::PointCloud<PointT>::ConstPtr& cloud) {
	PointCloudPtr cloud_filtered = preprocessCloud(cloud);
	// extract those planes that are considered as stairs and put them in cloud_stairs_
	findStairs(cloud_filtered);
	std::vector<pcl::PointIndices> cluster_indices = getStairClusters();
	return clustersToPointClouds(cluster_indices);

	return vec_cloud_stairs_;
//  std::vector<pcl::PointIndices> stair_cluster_indices = getStairClusters(cloud_stairs_);
//  return clustersToPointClouds(cloud_stairs_, stair_cluster_indices);
}

} // namespace lepp

#endif
