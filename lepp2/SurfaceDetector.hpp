#ifndef BASE_SURFACE_DETECTOR_H_
#define BASE_SURFACE_DETECTOR_H_

#include <vector>
#include <algorithm>

#include <pcl/visualization/cloud_viewer.h>

#include "lepp2/VideoObserver.hpp"
#include "lepp2/BaseSegmenter.hpp"
#include "lepp2/NoopSegmenter.hpp"
#include "lepp2/SurfaceSegmenter.hpp"
#include "lepp2/SurfaceAggregator.hpp"
#include "lepp2/ObjectApproximator.hpp"
#include "lepp2/MomentOfInertiaApproximator.hpp"
#include "lepp2/SplitApproximator.hpp"

using namespace lepp;

#include "lepp2/debug/timer.hpp"

template<class PointT>
class SurfaceDetector : public lepp::VideoObserver<PointT> {

 public:
    SurfaceDetector();
    virtual ~SurfaceDetector() {}

    /**
     * VideoObserver interface method implementation.
     */
    virtual void notifyNewFrame(
        int idx,
        const typename pcl::PointCloud<PointT>::ConstPtr& point_cloud);
    /**
       * Attaches a new SurfaceAggregator, which will be notified of newly detected
       * surfaces by this detector.
       */
    void attachSurfaceAggregator(boost::shared_ptr<SurfaceAggregator<PointT> > aggregator);

  protected:
    /// Helper typedefs to make the implementation cleaner
    typedef pcl::PointCloud<PointT> PointCloudT;
    typedef typename PointCloudT::Ptr PointCloudPtr;
    typedef typename PointCloudT::ConstPtr CloudConstPtr;

    /**
     * Notifies any observers about newly detected surfaces.
     */
    void notifySurfaces(std::vector<CloudConstPtr> surfaces);

  private:

    typename pcl::PointCloud<PointT>::ConstPtr cloud_;

    /**
     * Tracks all attached SurfaceAggregators that wish to be notified of newly
     * detected surfaces.
     */
    std::vector<boost::shared_ptr<SurfaceAggregator<PointT> > > aggregators;

    boost::shared_ptr<BaseSegmenter<PointT> > segmenter_;
    boost::shared_ptr<ObjectApproximator<PointT> > approximator_;

    /**
     * Performs a new update of the surface approximations.
     * Triggered when the detector is notified of a new frame (i.e. point cloud).
     */
    void update();
};

template<class PointT>
SurfaceDetector<PointT>::SurfaceDetector()
    : approximator_(new SplitObjectApproximator<PointT>(
        boost::shared_ptr<ObjectApproximator<PointT> >(
          new MomentOfInertiaObjectApproximator<PointT>))),
      segmenter_(new SurfaceSegmenter<PointT>()) {
}

template<class PointT>
void SurfaceDetector<PointT>::notifyNewFrame(
    int id,
    const typename pcl::PointCloud<PointT>::ConstPtr& point_cloud) {
  cloud_ = point_cloud;
  try {
    update();
  } catch (...) {
    std::cerr << "SurfaceDetector: Surface detection failed ..." << std::endl;
  }
}

template<class PointT>
void SurfaceDetector<PointT>::update() {

  Timer t;
  t.start();
  std::vector<CloudConstPtr> surfaces(segmenter_->segment(cloud_));
  t.stop();
  std::cerr << "Surface segmentation took " << t.duration() << std::endl;

  notifySurfaces(surfaces);

//  // Iteratively approximate the segments
//  size_t segment_count = segments.size();
//  std::vector<ObjectModelPtr> models;
//  for (size_t i = 0; i < segment_count; ++i) {
//    models.push_back(approximator_->approximate(segments[i]));
//  }

//  std::cerr << "Obstacle detection took " << t.duration() << std::endl;
//
//  notifyObstacles(models);
}

template<class PointT>
void SurfaceDetector<PointT>::attachSurfaceAggregator(
    boost::shared_ptr<SurfaceAggregator<PointT> > aggregator) {
  aggregators.push_back(aggregator);
}

template<class PointT>
void SurfaceDetector<PointT>::notifySurfaces(std::vector<CloudConstPtr> surfaces) {
  size_t sz = aggregators.size();
  for (size_t i = 0; i < sz; ++i) {
    aggregators[i]->updateSurfaces(surfaces);
  }
}

#endif
