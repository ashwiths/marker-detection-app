const aruco = require('js-aruco');
console.log(Object.keys(aruco));
// see if CV or POS3D or something else is exported
if (aruco.CV) console.log(Object.keys(aruco.CV));
if (aruco.SVD) console.log('SVD exported');
