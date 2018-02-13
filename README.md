# Computer Vision
Homework #2

Due: Tu 02/27/18 11:59 PM
See Homework - 2 description file (HW2.pdf) for details.

Create data points:
(1 Pts.) Write code to create the data points

  - Starter code available in directory fitting/
  - fitting/models.py: Put your code in "create_dataPoints". You are welcome to add more function or add nested functions with in the function.
  - Make sure the function returns and object with data points (List, tuple,...)
  - For this part of the assignment, please implement your own code for all computations, do not use inbuilt functions like from numpy, opencv or other libraries for clustering or segmentation.
  - Describe your method and findings in your report
  - This part of the assignment can be run using cv_hw1.py (there is no need to edit this file)
  - Usage: 
  
            ./cv_hm2 -i1 image1 -i2 image2 -m TL -t 20
  
            python cv_hm2.py -i1 image1 -i2 image2 -m TL -t 20
  - Please make sure your code runs when you run the above command from prompt/terminal
  - Any output images or files must be saved to "output/" folder (cv_hm2.py automatically does this)
  
-------------
Plot data points:
(1 Pts.) Write code to plot the data points on an image

  - Starter code available in directory fitting/
  - fitting/models.py: Put your code in "plot_data". You are welcome to add more function or add nested functions with in the function.
  - Make sure the function returns an image with datapoints plotted
  - For this part of the assignment, please implement your own code for all computations, do not use inbuilt functions like from numpy, opencv or other libraries for clustering or segmentation.
  - Describe your method and findings in your report
  - This part of the assignment can be run using cv_hm2.py (there is no need to edit this file)
  - Usage: 
  
            ./cv_hm2 -i1 image1 -i2 image2 -m TL -t 20
  
            python cv_hm2.py -i1 image1 -i2 image2 -m TL -t 20
  - Please make sure your code runs when you run the above command from prompt/terminal
  - Any output images or files must be saved to "output/" folder (cv_hm2.py automatically does this)
  
-------------
Total least square fitting:
(1 Pts.) Write code to fit a line using total least square, identify outliers and segmented image with the changes pixels.
  - Starter code available in directory fitting/
  - fitting/models.py: Put your code in "fit_line_tls". You are welcome to add more function or add nested functions with in the function.
  - Make sure the function returns a tuple of three images (fitted line, thresholded image and segmented image) 
  - For this part of the assignment, please implement your own code for all computations, do not use inbuilt functions like from numpy, opencv or other libraries for clustering or segmentation.
  - Describe your method and findings in your report
  - This part of the assignment can be run using cv_hm2.py (there is no need to edit this file)
  - Usage: 
  
            ./cv_hm2 -i1 image1 -i2 image2 -m TL -t 20
  
            python cv_hm2.py -i1 image1 -i2 image2 -m TL -t 20
  - Please make sure your code runs when you run the above command from prompt/terminal
  - Any output images or files must be saved to "output/" folder (cv_hm2.py automatically does this)
  
 -------------
 Robust estimation:
(1 Pts.) Write code to fit a line using robust estimators, identify outliers and segmented image with the changes pixels.
  - Starter code available in directory fitting/
  - fitting/models.py: Put your code in "fit_line_robust". You are welcome to add more function or add nested functions with in the function.
  - Make sure the function returns a tuple of three images (fitted line, thresholded image and segmented image) 
  - For this part of the assignment, please implement your own code for all computations.
  - Exception, You are welcome to use the fitLinea function from opencv
  - Describe your method and findings in your report
  - This part of the assignment can be run using cv_hm2.py (there is no need to edit this file)
  - Usage: 
  
            ./cv_hm2 -i1 image1 -i2 image2 -m RO -t 20
  
            python cv_hm2.py -i1 image1 -i2 image2 -m RO -t 20
  - Please make sure your code runs when you run the above command from prompt/terminal
  - Any output images or files must be saved to "output/" folder (cv_hm2.py automatically does this)
  
  -------------
 Gaussian Fitting:
(1 Pts.) Write code to fit a gaussian model, identify outliers and segmented image with the changes pixels.
  - Starter code available in directory fitting/
  - fitting/models.py: Put your code in "fit_gaussian". You are welcome to add more function or add nested functions with in the function.
  - Make sure the function returns a tuple of three images (fitted line, thresholded image and segmented image) 
  - For this part of the assignment, please implement your own code for all computations, do not use inbuilt functions like from numpy, opencv or other libraries for clustering or segmentation.
  - Describe your method and findings in your report
  - This part of the assignment can be run using cv_hm2.py (there is no need to edit this file)
  - Usage: 
  
            ./cv_hm2 -i1 image1 -i2 image2 -m GA -t 0.7
  
            python cv_hm2.py -i1 image1 -i2 image2 -m GA -t 0.7
  - Please make sure your code runs when you run the above command from prompt/terminal
  - Any output images or files must be saved to "output/" folder (cv_hm2.py automatically does this)
  
  
