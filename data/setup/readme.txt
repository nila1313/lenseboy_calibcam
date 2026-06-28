Run calbration from https://github.com/bbo-lab/calibcam with
python -m calibcam --videos test_8cams10.ccv.mp4 test_8cams3.ccv.mp4 test_8cams4.ccv.mp4 test_8cams5.ccv.mp4 test_8cams6.ccv.mp4 test_8cams7.ccv.mp4 test_8cams8.ccv.mp4 --board bboboard-v2.npy

There is a viewer available at https://github.com/bbo-lab/calipy/tree/board_extension that you can start with
python cali.py --videos  test_8cams10.ccv.mp4 test_8cams3.ccv.mp4 test_8cams4.ccv.mp4 test_8cams5.ccv.mp4 test_8cams6.ccv.mp4 test_8cams8.ccv.mp4  --calib_file multicam_calibration.npy
after calibration.

You might also find https://github.com/bbo-lab/calibcamlib helpful, which is a library to work with the camera calibrations.
