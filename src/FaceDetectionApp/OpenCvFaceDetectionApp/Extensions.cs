using System;
using OpenCvSharp;
using OpenCvSharp.Dnn;

namespace OpenCvFaceDetectionApp
{
    public static class Extensions
    {
        const string model = "wwwroot\\data\\res10_300x300_ssd_iter_140000_fp16.caffemodel";
        const string config = "wwwroot\\data\\deploy.prototxt";
        static readonly Net faceNet = CvDnn.ReadNetFromCaffe(config, model);
        public static string ToBase64String(this Mat mat) =>
            Convert.ToBase64String(mat.ToBytes());

        public static Mat DetectFaceWithOpenCvDNN(this Mat sourceImage)
        {
            using var frame = sourceImage;
            Mat result = new Mat();
            int frameHeight = frame.Rows;
            int frameWidth = frame.Cols;

           // using var faceNet = CvDnn.ReadNetFromCaffe(config, model);
            using var blob = CvDnn.BlobFromImage(frame, 1.0, new Size(300, 300), new Scalar(104, 117, 123), false, false);
            faceNet.SetInput(blob, "data");

            using var detection = faceNet.Forward("detection_out");
            using var detectionMat = new Mat(detection.Size(2), detection.Size(3), MatType.CV_32F, detection.Ptr(0));

            for (int i = 0; i < detectionMat.Rows; i++)
            {
                float confidence = detectionMat.At<float>(i, 2);

                if (confidence > 0.7)
                {
                    int x1 = (int)(detectionMat.At<float>(i, 3) * frameWidth);
                    int y1 = (int)(detectionMat.At<float>(i, 4) * frameHeight);
                    int x2 = (int)(detectionMat.At<float>(i, 5) * frameWidth);
                    int y2 = (int)(detectionMat.At<float>(i, 6) * frameHeight);

                    Cv2.Rectangle(frame, new Point(x1, y1), new Point(x2, y2), new Scalar(0, 255, 0), 2, LineTypes.Link4);
                    frame.CopyTo(result);
                }
            }
            
            return result;
        }

        public static Mat DetectFaceWithOpenCv(this Mat sourceImage, CascadeClassifier haarCascade)
        {
            using var image = sourceImage;
            Mat result = new Mat();
            Mat grayImage = new Mat();
            Cv2.CvtColor(image, grayImage, ColorConversionCodes.BGRA2GRAY);
            Cv2.EqualizeHist(grayImage, grayImage);

            var faces = haarCascade.DetectMultiScale(
                image: grayImage,
                scaleFactor: 1.08,
                minNeighbors: 2,
                flags: HaarDetectionType.ScaleImage,
                minSize: new Size(30, 30));

            foreach (var face in faces)
            {
                Cv2.Rectangle(image, face, Scalar.Red, 2);
                image.CopyTo(result);
            }
            return result;
        }

    }
}
