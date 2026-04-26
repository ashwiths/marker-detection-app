const aruco = require('js-aruco');
const { AR, CV } = aruco;

// Create dummy RGBA
const w = 600, h = 600;
const rawData = new Uint8Array(w * h * 4);

const imageData = { width: w, height: h, data: rawData };
const detector = new AR.Detector();

// Intercept candidates
let foundCandidates = [];
detector.findMarkers = function(image, candidates) {
  foundCandidates = candidates;
  return [];
};

detector.detect(imageData);
console.log(foundCandidates.length);

// Warp grayscale
if (foundCandidates.length > 0) {
   const candidate = foundCandidates[0];
   const warpedDst = new CV.Image(100, 100);
   CV.warp(detector.grey, warpedDst, candidate, 100);
   console.log(warpedDst.data.length); // 10000
}
