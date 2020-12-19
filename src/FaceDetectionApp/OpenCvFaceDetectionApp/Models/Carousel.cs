using System.Collections.Generic;

namespace OpenCvFaceDetectionApp.Models
{
    public class Carousel
    {
        public string Id { get; set; }
        public List<CarouselItem> Items { get; set; }


    }

    public class CarouselItem
    {
        public string ImageSource { get; set; }
        public string Label { get; set; }
        public string Description { get; set; }
    }
}
