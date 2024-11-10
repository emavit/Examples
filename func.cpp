Mat flippingMatcher (Ptr<Feature2D> sift, Mat patch, Mat descriptorsImage,  float distanceRatio, int minDis){
    // Variable to consider the best number of matches find between
    int numMatchesBest = -1;
    // Vector to store the output image
    Mat imageOut;
    // Loop over the 4 possible flipping of the patch, including the original one. Because of the 'flip' option, we need to start from -1
    for (int i=-1; i<=2; i++){
        vector<KeyPoint> kpTemp; 
        Mat descTemp;
        vector<DMatch> matches;
        Mat flippedPatch;
        // Flip the image. The first 3 iterations are done with different flipping, the fourth one with the original image
        if (i<=1){
            flip(patch, flippedPatch, i);
        }
        else{
            flippedPatch = patch.clone();
        }
        // Detect and compute keypoints and descriptors of the flipped patch
        tie (descTemp, kpTemp) = KeypointsFeatureExtractor(sift, flippedPatch, false,1);
        // Find matches between the patch and the descriptors passed as input
        matches = matchRefine(descriptorsImage, descTemp, distanceRatio, Personal_Flags::SIFT, minDis);
        // Update the best patch at each iteration
        if(int(matches.size())>numMatchesBest){
            numMatchesBest = int(matches.size());
            imageOut = flippedPatch.clone();
        }
    }
    return imageOut;
}