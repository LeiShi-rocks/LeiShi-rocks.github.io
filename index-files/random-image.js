// Array of image paths
const images = [
    "index-files/IMG-4803.jpeg",
    "index-files/IMG-4948.jpeg",
    "index-files/DSC02026.JPG",
    "index-files/DSC01882.JPG",
    "index-files/DSC01790.JPG",
    "index-files/DSC01792.JPG",
    "index-files/DSC01795.JPG",
//    "index-files/IMG-00130.JPG",
//    "index-files/IMG-0852.JPG",
    "index-files/DSC01795.JPG"
//    "index-files/Genentech-half.jpg"
];

// Function to select a random image and set it as the source
function setRandomImage() {
    const randomIndex = Math.floor(Math.random() * images.length);
    const imgElement = document.getElementById('randomImage');
    imgElement.src = images[randomIndex];
}

// Run the function after the page loads
window.onload = setRandomImage;
