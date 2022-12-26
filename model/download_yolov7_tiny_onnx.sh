FILE_ID=1-P3RpFnbXv_a99EW_wiBCUcZ7ssJzEnj
FILE_NAME=yolov7-tiny.onnx
CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=${FILE_ID}" -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1/p');
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" -O ${FILE_NAME};
rm -f /tmp/cookies.txt
