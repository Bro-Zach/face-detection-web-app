using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Forms;
using OpenCvSharp;
using OpenCvSharp.Dnn;

namespace OpenCvFaceDetectionApp.Pages
{
    public class IndexBase : ComponentBase
    {
        readonly CascadeClassifier _haarCascade = new CascadeClassifier("wwwroot\\haarcascade_frontalface_alt.xml");
        public string imgSrc;
        public string imgSrcDnn;
        public string error;

        //protected override Task OnInitializedAsync()
        //{
        //    faceNet = CvDnn.ReadNetFromCaffe(config, model);
        //    return base.OnInitializedAsync();
        //}

        public async Task HandleFileChange(InputFileChangeEventArgs args)
        {
            error = string.Empty;
            var format = "image/png";
            string contentType = args.File.ContentType;
            if (contentType.StartsWith("image/"))
            {
                Mat imageCopy = new Mat();
                Mat sourceImage = await ReadImageFromFile(args.File, format);
                sourceImage.CopyTo(imageCopy);
                DetectFaceWithOpenCv(contentType, sourceImage);
                DetectFaceWithOpenCvDNN(imageCopy);
            }
            else
            {
                error = $"({args.File.Name}) is not a valid image file. ";
                imgSrc = string.Empty;
            }
        }

        private void DetectFaceWithOpenCvDNN(Mat sourceImage)
        {
           
            var frame = sourceImage.DetectFaceWithOpenCvDNN();

            imgSrcDnn = $"data:image/png;base64,{frame.ToBase64String()}";

        }

        private void DetectFaceWithOpenCv(string contentType, Mat sourceImage)
        {
            using var image = sourceImage;
            Mat grayImage = new Mat();
            Cv2.CvtColor(image, grayImage, ColorConversionCodes.BGRA2GRAY);
            Cv2.EqualizeHist(grayImage, grayImage);

            var faces = _haarCascade.DetectMultiScale(
                image: grayImage,
                scaleFactor: 1.08,
                minNeighbors: 2,
                flags: HaarDetectionType.ScaleImage,
                minSize: new Size(30, 30));

            foreach (var face in faces)
            {
                Cv2.Rectangle(image, face, Scalar.Red, 2);
            }
            //using var result = sourceImage.DetectFaceWithOpenCv(_haarCascade);
            imgSrc = $"data:{contentType};base64,{image.ToBase64String()}";
        }

        private static async Task<Mat> ReadImageFromFile(IBrowserFile file, string format)
        {
            IBrowserFile imageFile = await file.RequestImageFileAsync(format, 500, 500);

            var buffer = new byte[imageFile.Size];
            await imageFile.OpenReadStream().ReadAsync(buffer);

            Mat sourceImage = Mat.FromImageData(buffer);
            return sourceImage;
        }
    }
}
