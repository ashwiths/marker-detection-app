import React, {useEffect, useRef, useState} from 'react';
import {
  View,
  Text,
  Image,
  ActivityIndicator,
  ScrollView,
  Dimensions,
  TouchableOpacity,
  StyleSheet,
} from 'react-native';
import {Camera, useCameraDevices} from 'react-native-vision-camera';
import RNFS from 'react-native-fs';
import ImageResizer from 'react-native-image-resizer';
import jpeg from 'jpeg-js';
import {Buffer} from 'buffer';

// Polyfill Buffer globally so jpeg-js can find it at runtime
if (typeof global.Buffer === 'undefined') {
  (global as any).Buffer = Buffer;
}

const {width: SCREEN_W, height: SCREEN_H} = Dimensions.get('window');

const MAX_MARKERS = 20;

// ── Scan box in screen-space ─────────────────────────────────────────────────
const BOX_W = SCREEN_W * 0.85;
const BOX_H = SCREEN_H * 0.48;
const BOX_LEFT = (SCREEN_W - BOX_W) / 2;
const BOX_TOP = (SCREEN_H - BOX_H) / 2;

// ── Types ─────────────────────────────────────────────────────────────────────
type Rotation = 0 | 90 | 180 | 270;

interface DetectionResult {
  absMinX: number;
  absMinY: number;
  absMaxX: number;
  absMaxY: number;
  rotation: Rotation;
}

export default function App() {
  const devices = useCameraDevices();
  const device = devices.back;
  const cameraRef = useRef<Camera>(null);

  const [croppedImagePath, setCroppedImagePath] = useState<string | null>(null);
  const [detected, setDetected] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [statusMsg, setStatusMsg] = useState('Point camera at marker and press Capture');
  const [markerList, setMarkerList] = useState<{uri: string; rot: Rotation}[]>([]);

  useEffect(() => {
    Camera.requestCameraPermission();
  }, []);

  // ───────────────────────────────────────────────────────────────────────────
  // PURE-JS PIXEL ROTATION
  // Rotates an RGBA buffer by 0/90/180/270 degrees clockwise.
  // ───────────────────────────────────────────────────────────────────────────
  const rotateBuffer = (
    data: Uint8Array,
    w: number,
    h: number,
    rotation: Rotation,
  ): {data: Uint8Array; width: number; height: number} => {
    if (rotation === 0) {
      return {data, width: w, height: h};
    }

    const outW = rotation === 90 || rotation === 270 ? h : w;
    const outH = rotation === 90 || rotation === 270 ? w : h;
    const out = new Uint8Array(outW * outH * 4);

    for (let row = 0; row < h; row++) {
      for (let col = 0; col < w; col++) {
        const srcOff = (row * w + col) * 4;
        let destRow: number;
        let destCol: number;

        if (rotation === 90) {
          // 90° clockwise: (row, col) → (col, h-1-row)
          destRow = col;
          destCol = h - 1 - row;
        } else if (rotation === 180) {
          // 180°: (row, col) → (h-1-row, w-1-col)
          destRow = h - 1 - row;
          destCol = w - 1 - col;
        } else {
          // 270° clockwise: (row, col) → (w-1-col, row)
          destRow = w - 1 - col;
          destCol = row;
        }

        const destOff = (destRow * outW + destCol) * 4;
        out[destOff] = data[srcOff];
        out[destOff + 1] = data[srcOff + 1];
        out[destOff + 2] = data[srcOff + 2];
        out[destOff + 3] = data[srcOff + 3];
      }
    }

    return {data: out, width: outW, height: outH};
  };

// ───────────────────────────────────────────────────────────────────────────
  // Helper: Warp RGBA using a homography matrix
  // ───────────────────────────────────────────────────────────────────────────
  const warpRGBA = (srcData: Uint8Array, srcW: number, srcH: number, dstW: number, dstH: number, m: any): Uint8Array => {
    const dst = new Uint8Array(dstW * dstH * 4);
    let pos = 0;
    const r_base = m[8], s_base = m[2], t_base = m[5];

    for (let i = 0; i < dstH; ++i) {
      let r = r_base + m[7] * i;
      let s = s_base + m[1] * i;
      let t = t_base + m[4] * i;

      for (let j = 0; j < dstW; ++j) {
        const u = r + m[6] * j;
        const v = s + m[0] * j;
        const w = t + m[3] * j;

        const x = v / u;
        const y = w / u;

        // Bilinear interpolation
        const sx1 = x >>> 0;
        const sx2 = (sx1 === srcW - 1) ? sx1 : sx1 + 1;
        const dx1 = x - sx1;
        const dx2 = 1.0 - dx1;

        const sy1 = y >>> 0;
        const sy2 = (sy1 === srcH - 1) ? sy1 : sy1 + 1;
        const dy1 = y - sy1;
        const dy2 = 1.0 - dy1;

        const p1 = (sy1 * srcW + sx1) * 4;
        const p2 = (sy1 * srcW + sx2) * 4;
        const p3 = (sy2 * srcW + sx1) * 4;
        const p4 = (sy2 * srcW + sx2) * 4;

        for (let c = 0; c < 4; c++) {
          dst[pos++] =
            (dy2 * (dx2 * srcData[p1 + c] + dx1 * srcData[p2 + c]) +
             dy1 * (dx2 * srcData[p3 + c] + dx1 * srcData[p4 + c])) & 0xff;
        }
      }
    }
    return dst;
  };

  // ───────────────────────────────────────────────────────────────────────────
  // DETECTION PIPELINE
  // Returns null if no valid marker found, or warped 300x300 upright RGBA buffer
  // ───────────────────────────────────────────────────────────────────────────
  const runDetection = (
    rawData: Uint8Array,
    W: number,
    H: number,
  ): { data: Uint8Array; width: number; height: number; rotation: Rotation } | null => {

    // Step A: Define scan region
    const sX = Math.floor(W * 0.1);
    const eX = Math.floor(W * 0.9);
    const sY = Math.floor(H * 0.18);
    const eY = Math.floor(H * 0.82);
    const cW = eX - sX;
    const cH = eY - sY;

    // Step B: Extract scan region into its own buffer
    const scanBuf = new Uint8Array(cW * cH * 4);
    for (let row = 0; row < cH; row++) {
      const src = ((sY + row) * W + sX) * 4;
      const dst = row * cW * 4;
      for (let i = 0; i < cW * 4; i++) {
        scanBuf[dst + i] = rawData[src + i];
      }
    }

    // Step C: Run js-aruco Candidate Detection
    const aruco = require('js-aruco');
    const detector = new aruco.AR.Detector();
    let candidates: any[] = [];
    
    // Intercept findMarkers to grab all valid quadrilaterals before it rejects our custom markers
    detector.findMarkers = function(image: any, foundCandidates: any[]) {
      candidates = foundCandidates;
      return [];
    };

    detector.detect({ width: cW, height: cH, data: scanBuf });
    console.log(`[Detection] Found ${candidates.length} quadrilaterals in scan region`);

    // We warp each candidate to a perfect 100x100 straight square for analysis
    const WARP_SIZE = 100;
    
    for (let i = 0; i < candidates.length; i++) {
      const candidate = candidates[i];
      const warpedDst = new aruco.CV.Image(WARP_SIZE, WARP_SIZE);
      aruco.CV.warp(detector.grey, warpedDst, candidate, WARP_SIZE);
      
      const scanBr = (col: number, row: number): number => {
        return warpedDst.data[row * WARP_SIZE + col];
      };

      // Step D: Classify Marker 1 vs Marker 2
      const edgeMargin = 0.05;
      const sampleEdge = (isVertical: boolean, isFar: boolean): number => {
        let sum = 0, n = 0;
        if (isVertical) {
          const col = Math.floor((isFar ? 1 - edgeMargin : edgeMargin) * WARP_SIZE);
          for (let r = 15; r < 85; r++) { sum += scanBr(col, r); n++; }
        } else {
          const row = Math.floor((isFar ? 1 - edgeMargin : edgeMargin) * WARP_SIZE);
          for (let c = 15; c < 85; c++) { sum += scanBr(c, row); n++; }
        }
        return n > 0 ? sum / n : 255;
      };

      const topEdgeBr = sampleEdge(false, false);
      const bottomEdgeBr = sampleEdge(false, true);
      const leftEdgeBr = sampleEdge(true, false);
      const rightEdgeBr = sampleEdge(true, true);

      const EDGE_SOLID_THRESH = 100;
      const isTopSolid = topEdgeBr < EDGE_SOLID_THRESH;
      const isBottomSolid = bottomEdgeBr < EDGE_SOLID_THRESH;
      const isLeftSolid = leftEdgeBr < EDGE_SOLID_THRESH;
      const isRightSolid = rightEdgeBr < EDGE_SOLID_THRESH;

      const solidEdgesCount = [isTopSolid, isBottomSolid, isLeftSolid, isRightSolid].filter(Boolean).length;
      
      let rotation: Rotation = 0;
      let valid = false;

      if (solidEdgesCount >= 3) {
        // Marker 1 check
        const cPad = 0.10;
        const cSizeSmall = 0.13;
        const cSizeLarge = 0.30;

        const samplePatch = (baseRow: number, baseCol: number, patchSize: number): number => {
          const r0 = Math.floor(baseRow * WARP_SIZE);
          const c0 = Math.floor(baseCol * WARP_SIZE);
          const rEnd = Math.floor(r0 + patchSize * WARP_SIZE);
          const cEnd = Math.floor(c0 + patchSize * WARP_SIZE);
          let sum = 0, n = 0;
          for (let r = r0; r < rEnd; r++) {
            for (let c = c0; c < cEnd; c++) { sum += scanBr(c, r); n++; }
          }
          return n > 0 ? sum / n : 255;
        };

        const tlBr = samplePatch(cPad, cPad, cSizeSmall);
        const trBr = samplePatch(cPad, 1 - cPad - cSizeSmall, cSizeSmall);
        const brBr = samplePatch(1 - cPad - cSizeSmall, 1 - cPad - cSizeSmall, cSizeSmall);
        const blBr = samplePatch(1 - cPad - cSizeSmall, cPad, cSizeSmall);

        const CORNER_DARK_THRESH = 110;
        const darkCornerCount = [tlBr < CORNER_DARK_THRESH, trBr < CORNER_DARK_THRESH, brBr < CORNER_DARK_THRESH, blBr < CORNER_DARK_THRESH].filter(Boolean).length;

        if (darkCornerCount === 1) {
          let largeBr = 255;
          if (tlBr < CORNER_DARK_THRESH) { largeBr = samplePatch(cPad, cPad, cSizeLarge); rotation = 0; }
          else if (trBr < CORNER_DARK_THRESH) { largeBr = samplePatch(cPad, 1 - cPad - cSizeLarge, cSizeLarge); rotation = 90; }
          else if (brBr < CORNER_DARK_THRESH) { largeBr = samplePatch(1 - cPad - cSizeLarge, 1 - cPad - cSizeLarge, cSizeLarge); rotation = 180; }
          else if (blBr < CORNER_DARK_THRESH) { largeBr = samplePatch(1 - cPad - cSizeLarge, cPad, cSizeLarge); rotation = 270; }

          if (largeBr >= 140) {
            valid = true;
          }
        }
      } else if (solidEdgesCount === 2) {
        // Marker 2 check
        if (isLeftSolid && isBottomSolid) { rotation = 0; valid = true; }
        else if (isTopSolid && isLeftSolid) { rotation = 90; valid = true; }
        else if (isTopSolid && isRightSolid) { rotation = 180; valid = true; }
        else if (isRightSolid && isBottomSolid) { rotation = 270; valid = true; }
      }

      if (valid) {
        console.log(`[Detection] Success! Found valid marker candidate. Rotation: ${rotation}°`);
        
        // Convert candidate corners from scanBuf space to raw image space
        const absCorners = candidate.map((p: any) => ({ x: p.x + sX, y: p.y + sY }));
        
        // Get homography matrix for 300x300 output
        const OUT_SIZE = 300;
        const m = aruco.CV.getPerspectiveTransform(absCorners, OUT_SIZE - 1);
        
        // Warp raw RGBA to perfect 300x300 square
        let finalBuf = warpRGBA(rawData, W, H, OUT_SIZE, OUT_SIZE, m);
        
        // Apply final JS rotation so it's upright
        finalBuf = rotateBuffer(finalBuf, OUT_SIZE, OUT_SIZE, rotation).data;
        
        return { data: finalBuf, width: OUT_SIZE, height: OUT_SIZE, rotation };
      }
    }

    console.log('[Detection] No valid marker passed validation.');
    return null;
  };

  // ───────────────────────────────────────────────────────────────────────────
  // MAIN DETECT FUNCTION
  // ───────────────────────────────────────────────────────────────────────────
  const detectMarker = async (imagePath: string) => {
    const t0 = Date.now();

    try {
      // 1. Resize to 600×600 for faster JS processing while keeping good detail
      const resized = await ImageResizer.createResizedImage(
        imagePath,
        600,
        600,
        'JPEG',
        85,
        0,
        undefined,
        false,
        {mode: 'contain'},
      );
      console.log(`[Pipeline] Resized in ${Date.now() - t0}ms → ${resized.width}x${resized.height}`);

      // 2. Decode JPEG → raw RGBA pixel buffer
      const b64 = await RNFS.readFile(resized.uri, 'base64');
      const raw = jpeg.decode(Buffer.from(b64, 'base64'), {useTArray: true});
      console.log(`[Pipeline] Decoded ${raw.width}x${raw.height} in ${Date.now() - t0}ms`);

      // 3. Run Candidate Quad Detection + Warping + Classification
      const result = runDetection(raw.data as Uint8Array, raw.width, raw.height);

      if (!result) {
        setStatusMsg('❌ No Marker Found');
        setDetected(false);
        setIsProcessing(false);
        return;
      }

      // result.data is already the perfectly straight, upright, 300x300 warped RGBA buffer
      const { data: finalBuf, width: OUT_W, height: OUT_H, rotation } = result;

      // 4. Encode directly to JPEG
      const encoded = jpeg.encode(
        {data: finalBuf, width: OUT_W, height: OUT_H},
        90,
      );
      const b64out = Buffer.from(encoded.data).toString('base64');

      // 5. Write to cache
      const finalPath = `${RNFS.CachesDirectoryPath}/marker_${Date.now()}.jpg`;
      await RNFS.writeFile(finalPath, b64out, 'base64');

      const elapsed = Date.now() - t0;
      console.log(`[Pipeline] ✅ Total: ${elapsed}ms, rotation: ${rotation}°`);

      setCroppedImagePath('file://' + finalPath);
      setDetected(true);
      setStatusMsg(`✅ Detected & Cropped! (${elapsed}ms | rot: ${rotation}°)`);

      setMarkerList(prev => [{uri: 'file://' + finalPath, rot: rotation}, ...prev].slice(0, MAX_MARKERS));
    } catch (err: any) {
      const msg = err?.message ?? String(err);
      console.log('[Pipeline] Error:', msg, err);
      setStatusMsg(`⚠️ ${msg.substring(0, 100)}`);
      setDetected(false);
    } finally {
      setIsProcessing(false);
    }
  };

  // ───────────────────────────────────────────────────────────────────────────
  // CAPTURE
  // ───────────────────────────────────────────────────────────────────────────
  const takePhoto = async () => {
    if (!cameraRef.current) { return; }
    setIsProcessing(true);
    setDetected(false);
    setCroppedImagePath(null);
    setStatusMsg('📷 Capturing...');

    try {
      const photo = await cameraRef.current.takePhoto({
        flash: 'off',
        qualityPrioritization: 'quality',
      });
      const path = 'file://' + photo.path;
      console.log(`[Capture] Photo: ${photo.width}x${photo.height}`);
      await detectMarker(path);
    } catch (e) {
      console.log('[Capture] Error:', e);
      setIsProcessing(false);
      setStatusMsg('⚠️ Capture failed');
    }
  };

  // ───────────────────────────────────────────────────────────────────────────
  // RENDER
  // ───────────────────────────────────────────────────────────────────────────
  if (!device) {
    return (
      <View style={styles.center}>
        <Text style={styles.white}>Loading camera...</Text>
      </View>
    );
  }

  return (
    <View style={styles.root}>
      {/* Live camera feed */}
      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        photo={true}
      />

      {/* Dark overlay — top */}
      <View style={[styles.overlay, {top: 0, left: 0, right: 0, height: BOX_TOP}]} />
      {/* Dark overlay — bottom */}
      <View style={[styles.overlay, {top: BOX_TOP + BOX_H, left: 0, right: 0, bottom: 0}]} />
      {/* Dark overlay — left */}
      <View style={[styles.overlay, {top: BOX_TOP, left: 0, width: BOX_LEFT, height: BOX_H}]} />
      {/* Dark overlay — right */}
      <View style={[styles.overlay, {top: BOX_TOP, left: BOX_LEFT + BOX_W, right: 0, height: BOX_H}]} />

      {/* Green scan box */}
      <View style={[styles.scanBox, {top: BOX_TOP, left: BOX_LEFT, width: BOX_W, height: BOX_H}]}>
        {/* Corner accents */}
        <View style={[styles.corner, styles.cornerTL]} />
        <View style={[styles.corner, styles.cornerTR]} />
        <View style={[styles.corner, styles.cornerBL]} />
        <View style={[styles.corner, styles.cornerBR]} />
      </View>

      {/* Guide label */}
      <Text style={[styles.guide, {top: BOX_TOP + 15}]}>
        Align marker inside box
      </Text>

      {/* Status message */}
      <Text style={[styles.statusText, {bottom: markerList.length > 0 ? 225 : 125}]}>
        {statusMsg}
      </Text>

      {/* Cropped marker preview */}
      {croppedImagePath && (
        <View style={styles.previewCard}>
          <Text style={styles.previewLabel}>Extracted Marker (300×300)</Text>
          <Image source={{uri: croppedImagePath}} style={styles.previewImg} />
        </View>
      )}

      {/* Capture button */}
      <TouchableOpacity
        id="captureButton"
        style={[styles.captureBtn, {bottom: markerList.length > 0 ? 155 : 55}]}
        onPress={takePhoto}
        disabled={isProcessing}
        activeOpacity={0.8}>
        {isProcessing ? (
          <ActivityIndicator color="#00ff88" size="large" />
        ) : (
          <Text style={styles.captureText}>📷  Capture</Text>
        )}
      </TouchableOpacity>

      {/* Marker history gallery */}
      {markerList.length > 0 && (
        <View style={styles.gallery}>
          <Text style={styles.galleryTitle}>
            Marker History — {markerList.length} / {MAX_MARKERS}
          </Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.galleryScroll}>
            {markerList.map((item, i) => (
              <View key={i} style={styles.thumb}>
                <Image source={{uri: item.uri}} style={styles.thumbImg} />
                <Text style={styles.thumbLabel}>#{markerList.length - i}</Text>
                {item.rot !== 0 && (
                  <Text style={styles.thumbRot}>{item.rot}°</Text>
                )}
              </View>
            ))}
          </ScrollView>
        </View>
      )}
    </View>
  );
}

// ── Styles ───────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  root: {flex: 1, backgroundColor: '#000'},
  center: {flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#000'},
  white: {color: '#fff', fontSize: 16},

  overlay: {
    position: 'absolute',
    backgroundColor: 'rgba(0,0,0,0.58)',
  },

  scanBox: {
    position: 'absolute',
    borderWidth: 2,
    borderColor: '#00ff88',
    borderRadius: 2,
  },

  // Corner accent pieces
  corner: {
    position: 'absolute',
    width: 20,
    height: 20,
    borderColor: '#00ff88',
    borderWidth: 3,
  },
  cornerTL: {top: -2, left: -2, borderRightWidth: 0, borderBottomWidth: 0},
  cornerTR: {top: -2, right: -2, borderLeftWidth: 0, borderBottomWidth: 0},
  cornerBL: {bottom: -2, left: -2, borderRightWidth: 0, borderTopWidth: 0},
  cornerBR: {bottom: -2, right: -2, borderLeftWidth: 0, borderTopWidth: 0},

  guide: {
    position: 'absolute',
    alignSelf: 'center',
    color: '#00ff88',
    fontSize: 13,
    fontWeight: '600',
    backgroundColor: 'rgba(0,0,0,0.65)',
    paddingHorizontal: 12,
    paddingVertical: 3,
    borderRadius: 4,
  },

  statusText: {
    position: 'absolute',
    alignSelf: 'center',
    color: '#ffffff',
    fontSize: 13,
    fontWeight: 'bold',
    backgroundColor: 'rgba(0,0,0,0.75)',
    paddingHorizontal: 14,
    paddingVertical: 5,
    borderRadius: 8,
    textAlign: 'center',
    maxWidth: SCREEN_W * 0.9,
  },

  previewCard: {
    position: 'absolute',
    top: Math.max(30, BOX_TOP - 140),
    alignSelf: 'center',
    backgroundColor: 'rgba(10,10,20,0.85)',
    borderWidth: 1,
    borderColor: '#00ff88',
    borderRadius: 8,
    padding: 8,
    alignItems: 'center',
  },
  previewLabel: {color: '#aaa', fontSize: 11, marginBottom: 6},
  previewImg: {width: 90, height: 90, resizeMode: 'stretch'},

  captureBtn: {
    position: 'absolute',
    alignSelf: 'center',
    backgroundColor: 'rgba(0,10,5,0.9)',
    borderWidth: 2,
    borderColor: '#00ff88',
    paddingHorizontal: 40,
    paddingVertical: 14,
    borderRadius: 50,
    minWidth: 160,
    alignItems: 'center',
  },
  captureText: {
    color: '#00ff88',
    fontSize: 17,
    fontWeight: 'bold',
    letterSpacing: 0.5,
  },

  gallery: {
    position: 'absolute',
    bottom: 0,
    width: '100%',
    backgroundColor: 'rgba(5,5,15,0.92)',
    borderTopWidth: 1,
    borderTopColor: '#1a1a2e',
    paddingTop: 8,
    paddingBottom: 10,
  },
  galleryTitle: {
    color: '#555',
    fontSize: 10,
    fontWeight: 'bold',
    paddingHorizontal: 12,
    marginBottom: 6,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  galleryScroll: {paddingHorizontal: 8},
  thumb: {
    alignItems: 'center',
    marginHorizontal: 4,
    width: 72,
  },
  thumbImg: {
    width: 64,
    height: 64,
    resizeMode: 'cover',
    borderRadius: 4,
    borderWidth: 1,
    borderColor: '#222',
  },
  thumbLabel: {color: '#555', fontSize: 9, marginTop: 3},
  thumbRot: {
    color: '#00ff88',
    fontSize: 8,
    fontWeight: 'bold',
  },
});