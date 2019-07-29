#Lucas Kanade tracking

* Running example ' python3 lucas_kanade_tracking.py --dataset_dir ../images/Gym/img --rect 167,69,24,127'

##Investigation

* There are two different classes for tracking object via Lucas Kanade algorithm: LucasKanadeGoodFeatures and LucasKanadeFillRectPoints. Two of them used calcOpticalFlowPyrLK function from OpenCV. 
* LucasKanadeGoodFeatures use goodFeaturesToTrack from OpenCV to find feature points in the rectangle where object exist. Sometimes it gives realy good points from the object, but sometimes it return points from background and tracking loses
* LucasKanadeFillRectPoints use all points from object rectangle as input for algorithm 
* After calcOpticalFlowPyrLK retruned us potential points we tried to find shift related to previous poiints, so we find average delta x and delta y. 
After that we move our rectangle which track the object 
* As a result there were recorded two videos LK_GoodFeatures_Woman.mov and LK_rectangle_GYM.mov
* LK_GoodFeatures_Woman.mov was recorded using implementation form LucasKanadeGoodFeatures class.
Here we can notice that tracking was lost when good features detect tree and fit on it. So, the problem of this approch is that method goodFeaturesToTrack can give us points which are not from object
* LK_rectangle_GYM.mov was recordewd using LucasKanadeFillRectPoints implementation, it illustrate good result, maybe there are strong difference between object color and background color

#Template matching
* Example of running temmplate matching: <br>
' python3 -m Lucas-Kanade.template_matching --image images/Liquor/img/0001.jpg --template images/Liquor/template.jpg --method ssd --output images/Liquor/0001_template.jpg'



