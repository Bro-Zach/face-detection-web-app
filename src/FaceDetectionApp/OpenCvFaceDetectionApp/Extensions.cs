using System;
using OpenCvSharp;

namespace OpenCvFaceDetectionApp
{
    public static class Extensions
    {
        public static string ToBase64String(this Mat mat) =>
            Convert.ToBase64String(mat.ToBytes());
    }
}
