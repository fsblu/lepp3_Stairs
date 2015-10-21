#ifndef LEPP2_SPLIT_APPROXIMATOR_H__
#define LEPP2_SPLIT_APPROXIMATOR_H__
#include "lepp2/ObjectApproximator.hpp"
#include "lepp2/models/ObjectModel.h"

#include <deque>
#include <map>

#include <pcl/common/pca.h>
#include <pcl/common/common.h>

namespace lepp {

/**
 * An ABC that represents the strategy for splitting a point cloud used by the
 * `SplitObjectApproximator`.
 *
 * Varying the `SplitStrategy` implementation allows us to change how point
 * clouds are split (or if they are split at all) without changing the logic of
 * the `SplitObjectApproximator` approximator itself.
 */
template<class PointT>
class SplitStrategy {
public:
  /**
   * A pure virtual method that concrete implementations need to define.
   *
   * :param split_depth: The current split depth, i.e. the number of times the
   *     original cloud has already been split
   * :param point_cloud: The current point cloud that should be split by the
   *    `SplitStrategy` implementation.
   * :returns: The method should return a vector of point clouds obtained by
   *      splitting the given cloud into any number of parts. If the given
   *      point cloud should not be split, an empty vector should be returned.
   *      Once the empty vector is returned, the `SplitObjectApproximator` will
   *      stop the splitting process for that branch of the split tree.
   */
  virtual std::vector<typename pcl::PointCloud<PointT>::Ptr> split(
      int split_depth,
      const typename pcl::PointCloud<PointT>::ConstPtr& point_cloud) = 0;
};

template<class PointT>
class DepthLimitSplitStrategy : public SplitStrategy<PointT> {
public:
  DepthLimitSplitStrategy(int depth_limit) : limit_(depth_limit) {}
  std::vector<typename pcl::PointCloud<PointT>::Ptr> split(
      int split_depth,
      const typename pcl::PointCloud<PointT>::ConstPtr& point_cloud);
private:
  /**
   * A helper method that does the actual split, when needed.
   */
  std::vector<typename pcl::PointCloud<PointT>::Ptr> doSplit(
      const typename pcl::PointCloud<PointT>::ConstPtr& point_cloud);

  int const limit_;
};

template<class PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr>
DepthLimitSplitStrategy<PointT>::split(
    int split_depth,
    const typename pcl::PointCloud<PointT>::ConstPtr& point_cloud) {

  if (split_depth < limit_) {
    return this->doSplit(point_cloud);
  } else {
    return std::vector<typename pcl::PointCloud<PointT>::Ptr>();
  }
}

template<class PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr>
DepthLimitSplitStrategy<PointT>::doSplit(
    const typename pcl::PointCloud<PointT>::ConstPtr& point_cloud) {
  typedef pcl::PointCloud<PointT> PointCloud;
  typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;
  // Compute PCA for the input cloud
  pcl::PCA<PointT> pca;
  pca.setInputCloud(point_cloud);
  Eigen::Vector3f eigenvalues = pca.getEigenValues();
  Eigen::Matrix3f eigenvectors = pca.getEigenVectors();

  Eigen::Vector3d main_pca_axis = eigenvectors.col(0).cast<double>();

  // Compute the centroid
  Eigen::Vector4d centroid;
  pcl::compute3DCentroid(*point_cloud, centroid);

  /// The plane equation
  double d = (-1) * (
      centroid[0] * main_pca_axis[0] +
      centroid[1] * main_pca_axis[1] +
      centroid[2] * main_pca_axis[2]
  );

  // Prepare the two parts.
  std::vector<PointCloudPtr> ret;
  ret.push_back(PointCloudPtr(new pcl::PointCloud<PointT>()));
  ret.push_back(PointCloudPtr(new pcl::PointCloud<PointT>()));
  PointCloud& first = *ret[0];
  PointCloud& second = *ret[1];

  // Now divide the input cloud into two clusters based on the splitting plane
  size_t const sz = point_cloud->size();
  for (size_t i = 0; i < sz; ++i) {
    // Boost the precision of the points we are dealing with to make the
    // calculation more precise.
    PointT const& original_point = (*point_cloud)[i];
    Eigen::Vector3f const vector_point = original_point.getVector3fMap();
    Eigen::Vector3d const point = vector_point.cast<double>();
    // Decide on which side of the plane the current point is and add it to the
    // appropriate partition.
    if (point.dot(main_pca_axis) + d < 0.) {
      first.push_back(original_point);
    } else {
      second.push_back(original_point);
    }
  }

  // Return the parts in a vector, as expected by the interface...
  return ret;
}

/**
 * An approximator implementation that will generate an approximation by
 * splitting the given object into multiple parts. Each part approximation is
 * generated by delegating to a wrapped `ObjectApproximator` instance, allowing
 * clients to vary the algorithm used for approximations, while keeping the
 * logic of incrementally splitting up the object.
 */
template<class PointT>
class SplitObjectApproximator : public ObjectApproximator<PointT> {
public:
  /**
   * Create a new `SplitObjectApproximator` that will approximate each part by
   * using the given approximator instance.
   */
  SplitObjectApproximator(boost::shared_ptr<ObjectApproximator<PointT> > approx)
      : approximator_(approx),
        // TODO Allow for split strategy injection
        splitter_(new DepthLimitSplitStrategy<PointT>(1)) {}
  /**
   * `ObjectApproximator` interface method.
   */
  boost::shared_ptr<CompositeModel> approximate(
      const typename pcl::PointCloud<PointT>::ConstPtr& point_cloud);
private:
  /**
   * An `ObjectApproximator` used to generate approximations for object parts.
   */
  boost::shared_ptr<ObjectApproximator<PointT> > approximator_;
  /**
   * The strategy to be used for splitting point clouds.
   */
  boost::shared_ptr<SplitStrategy<PointT> > splitter_;
};

template<class PointT>
boost::shared_ptr<CompositeModel> SplitObjectApproximator<PointT>::approximate(
    const typename pcl::PointCloud<PointT>::ConstPtr& point_cloud) {
  boost::shared_ptr<CompositeModel> approx(new CompositeModel);
  typedef typename pcl::PointCloud<PointT>::ConstPtr PointCloudConstPtr;
  typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;
  std::deque<std::pair<int, PointCloudConstPtr> > queue;
  queue.push_back(std::make_pair(0, point_cloud));

  while (!queue.empty()) {
    int const depth = queue[0].first;
    PointCloudConstPtr const current_cloud = queue[0].second;
    queue.pop_front();

    // Delegates to the wrapped approximator for each part's approximation.
    ObjectModelPtr model = approximator_->approximate(current_cloud);
    // TODO Decide whether the model fits well enough for the current cloud.
    // For now we fix the number of split iterations.
    if (depth == 0) {
      // The approximation should be improved. Try doing it for the split clouds
      std::vector<PointCloudPtr> const splits = splitter_->split(depth, current_cloud);
      // Add each new split section into the queue as children of the current
      // node.
      for (size_t i = 0; i < splits.size(); ++i) {
        queue.push_back(std::make_pair(depth + 1, splits[i]));
      }
    } else {
      // Keep the approximation
      approx->addModel(model);
    }
  }

  return approx;
}

}  // namespace lepp
#endif
