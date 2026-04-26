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
  Modal,
} from 'react-native';
import {Camera, useCameraDevices} from 'react-native-vision-camera';
import RNFS from 'react-native-fs';
import ImageResizer from 'react-native-image-resizer';
import jpeg from 'jpeg-js';
import {Buffer} from 'buffer';

if (typeof global.Buffer === 'undefined') {
  (global as any).Buffer = Buffer;
}

const {width: SCREEN_W, height: SCREEN_H} = Dimensions.get('window');
const MAX_MARKERS = 20;

const BOX_W = SCREEN_W * 0.85;
const BOX_H = SCREEN_H * 0.48;
const BOX_LEFT = (SCREEN_W - BOX_W) / 2;
const BOX_TOP = (SCREEN_H - BOX_H) / 2;

type Rotation = 0 | 90 | 180 | 270;

export default function App() {
  const devices = useCameraDevices();
  const device = devices.back;
  const cameraRef = useRef<Camera>(null);

  // Find optimal format satisfying: min 2000x2000px, max 3000x3000px
  const format = React.useMemo(() => {
    if (!device) return undefined;
    for (const f of device.formats) {
      if (f.videoWidth >= 2000 && f.videoWidth <= 3000 && f.videoHeight >= 2000 && f.videoHeight <= 3000) {
        return f;
      }
    }
    // Fallback: closest highest res <= 3000
    const valid = device.formats.filter(f => f.videoWidth <= 3000 && f.videoHeight <= 3000);
    if (valid.length > 0) {
      return valid.sort((a, b) => (b.videoWidth * b.videoHeight) - (a.videoWidth * a.videoHeight))[0];
    }
    return [...device.formats].sort((a, b) => (b.videoWidth * b.videoHeight) - (a.videoWidth * a.videoHeight))[0];
  }, [device]);

  const [croppedImagePath, setCroppedImagePath] = useState<string | null>(null);
  const [detected, setDetected] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [statusMsg, setStatusMsg] = useState('Point camera at marker and press Capture');
  const [markerList, setMarkerList] = useState<{uri: string; rot: Rotation}[]>([]);

  useEffect(() => {
    Camera.requestCameraPermission();
  }, []);

  // ── Pixel-level rotation ──────────────────────────────────────────────────
  const rotateBuffer = (
    data: Uint8Array,
    w: number,
    h: number,
    rotation: Rotation,
  ): {data: Uint8Array; width: number; height: number} => {
    if (rotation === 0) return {data, width: w, height: h};
    const outW = rotation === 90 || rotation === 270 ? h : w;
    const outH = rotation === 90 || rotation === 270 ? w : h;
    const out = new Uint8Array(outW * outH * 4);
    for (let row = 0; row < h; row++) {
      for (let col = 0; col < w; col++) {
        const srcOff = (row * w + col) * 4;
        let dR: number, dC: number;
        if (rotation === 90)       { dR = col;       dC = h - 1 - row; }
        else if (rotation === 180) { dR = h - 1 - row; dC = w - 1 - col; }
        else                       { dR = w - 1 - col; dC = row; }
        const dOff = (dR * outW + dC) * 4;
        out[dOff] = data[srcOff]; out[dOff+1] = data[srcOff+1];
        out[dOff+2] = data[srcOff+2]; out[dOff+3] = data[srcOff+3];
      }
    }
    return {data: out, width: outW, height: outH};
  };

  // ── Perspective warp ──────────────────────────────────────────────────────
  const warpRGBA = (
    srcData: Uint8Array, srcW: number, srcH: number,
    dstW: number, dstH: number, m: any,
  ): Uint8Array => {
    const dst = new Uint8Array(dstW * dstH * 4);
    let pos = 0;
    for (let i = 0; i < dstH; ++i) {
      for (let j = 0; j < dstW; ++j) {
        const u = m[8] + m[7]*i + m[6]*j;
        const x = (m[2] + m[1]*i + m[0]*j) / u;
        const y = (m[5] + m[4]*i + m[3]*j) / u;
        const sx1 = Math.min(x >>> 0, srcW - 2);
        const sy1 = Math.min(y >>> 0, srcH - 2);
        const sx2 = sx1 + 1, sy2 = sy1 + 1;
        const dx1 = x - sx1, dx2 = 1 - dx1;
        const dy1 = y - sy1, dy2 = 1 - dy1;
        const p1=(sy1*srcW+sx1)*4, p2=(sy1*srcW+sx2)*4;
        const p3=(sy2*srcW+sx1)*4, p4=(sy2*srcW+sx2)*4;
        for (let c = 0; c < 4; c++) {
          dst[pos++] = (dy2*(dx2*srcData[p1+c]+dx1*srcData[p2+c]) +
                        dy1*(dx2*srcData[p3+c]+dx1*srcData[p4+c])) & 0xff;
        }
      }
    }
    return dst;
  };

  // ── Detection pipeline ────────────────────────────────────────────────────
  const runDetection = (
    rawData: Uint8Array, W: number, H: number,
  ): {data: Uint8Array; width: number; height: number; rotation: Rotation} | null => {

    // Scan region: central 80% of the image
    const sX = Math.floor(W * 0.08);
    const eX = Math.floor(W * 0.92);
    const sY = Math.floor(H * 0.10);
    const eY = Math.floor(H * 0.90);
    const cW = eX - sX;
    const cH = eY - sY;

    // Extract scan region
    const scanBuf = new Uint8Array(cW * cH * 4);
    for (let row = 0; row < cH; row++) {
      const src = ((sY + row) * W + sX) * 4;
      const dst = row * cW * 4;
      for (let i = 0; i < cW * 4; i++) scanBuf[dst + i] = rawData[src + i];
    }

    const aruco = require('js-aruco');
    const detector = new aruco.AR.Detector();
    let candidates: any[] = [];
    detector.findMarkers = function(_image: any, fc: any[]) {
      candidates = fc;
      return [];
    };

    // ── Morphological CLOSE (Min then Max) ──
    // Bridges gaps in dashed lines without growing the outer boundary of the marker.
    // We use a radius large enough to bridge the 20px white gaps in Marker 2.
    const radius = Math.max(10, Math.floor(cW * 0.035));
    
    const gray = new Uint8Array(cW * cH);
    for (let i = 0; i < cW * cH; i++) {
      gray[i] = Math.round((scanBuf[i*4] + scanBuf[i*4+1] + scanBuf[i*4+2]) / 3);
    }
    
    // 1. Min Filter (expands dark regions)
    const tmpMin = new Uint8Array(cW * cH);
    for (let r = 0; r < cH; r++) {
      for (let c = 0; c < cW; c++) {
        let mn = 255;
        for (let k = Math.max(0,c-radius); k <= Math.min(cW-1,c+radius); k++) {
          const v = gray[r*cW+k]; if (v < mn) mn = v;
        }
        tmpMin[r*cW+c] = mn;
      }
    }
    const minFiltered = new Uint8Array(cW * cH);
    for (let c = 0; c < cW; c++) {
      for (let r = 0; r < cH; r++) {
        let mn = 255;
        for (let k = Math.max(0,r-radius); k <= Math.min(cH-1,r+radius); k++) {
          const v = tmpMin[k*cW+c]; if (v < mn) mn = v;
        }
        minFiltered[r*cW+c] = mn;
      }
    }

    // 2. Max Filter (shrinks dark regions back, preserving bridges)
    const tmpMax = new Uint8Array(cW * cH);
    for (let r = 0; r < cH; r++) {
      for (let c = 0; c < cW; c++) {
        let mx = 0;
        for (let k = Math.max(0,c-radius); k <= Math.min(cW-1,c+radius); k++) {
          const v = minFiltered[r*cW+k]; if (v > mx) mx = v;
        }
        tmpMax[r*cW+c] = mx;
      }
    }
    const closed = new Uint8Array(cW * cH * 4);
    for (let c = 0; c < cW; c++) {
      for (let r = 0; r < cH; r++) {
        let mx = 0;
        for (let k = Math.max(0,r-radius); k <= Math.min(cH-1,r+radius); k++) {
          const v = tmpMax[k*cW+c]; if (v > mx) mx = v;
        }
        const idx = (r*cW+c)*4;
        closed[idx] = closed[idx+1] = closed[idx+2] = mx;
        closed[idx+3] = 255;
      }
    }

    detector.detect({width: cW, height: cH, data: closed});
    console.log(`[Detection] Candidates: ${candidates.length}`);

    // Original grayscale for sampling
    const origGrey = new aruco.CV.Image(cW, cH);
    for (let i = 0; i < cW * cH; i++) {
      origGrey.data[i] = gray[i];
    }

    const WARP_SIZE = 100;

    for (let i = 0; i < candidates.length; i++) {
      const cand = candidates[i];

      // Area filter — skip tiny noise
      let area = 0;
      for (let j = 0; j < 4; j++) {
        const p1 = cand[j], p2 = cand[(j+1)%4];
        area += p1.x*p2.y - p2.x*p1.y;
      }
      if (Math.abs(area / 2) < 5000) continue;

      const wDst = new aruco.CV.Image(WARP_SIZE, WARP_SIZE);
      aruco.CV.warp(origGrey, wDst, cand, WARP_SIZE);

      const px = (col: number, row: number) => wDst.data[row * WARP_SIZE + col];

      // Sample a band along each edge (inner 12% strip)
      // Returns darkFrac: fraction of the edge length that is dark.
      const sampleEdgeBand = (side: 'top'|'bottom'|'left'|'right'): { darkFrac: number, avgBr: number } => {
        let sum = 0, n = 0, darkCount = 0;
        const STRIP = 12; // pixels from edge inward
        const MID_S = 10, MID_E = 90; // central 80% of the edge
        for (let pos = MID_S; pos < MID_E; pos++) {
          let minV = 255;
          for (let d = 1; d <= STRIP; d++) {
            let v: number;
            if      (side === 'top')    v = px(pos, d);
            else if (side === 'bottom') v = px(pos, WARP_SIZE-1-d);
            else if (side === 'left')   v = px(d, pos);
            else                        v = px(WARP_SIZE-1-d, pos);
            if (v < minV) minV = v;
          }
          sum += minV; 
          n++;
          if (minV < 150) darkCount++; 
        }
        return { darkFrac: darkCount / n, avgBr: sum / n };
      };

      const topE    = sampleEdgeBand('top');
      const bottomE = sampleEdgeBand('bottom');
      const leftE   = sampleEdgeBand('left');
      const rightE  = sampleEdgeBand('right');

      const classifyEdge = (edge: {darkFrac: number}): 'SOLID' | 'DASHED' | 'NONE' => {
        if (edge.darkFrac > 0.80) return 'SOLID';
        if (edge.darkFrac >= 0.25) return 'DASHED';
        return 'NONE';
      };

      const topType    = classifyEdge(topE);
      const bottomType = classifyEdge(bottomE);
      const leftType   = classifyEdge(leftE);
      const rightType  = classifyEdge(rightE);

      const solidCount  = [topType, bottomType, leftType, rightType].filter(t => t === 'SOLID').length;
      const dashedCount = [topType, bottomType, leftType, rightType].filter(t => t === 'DASHED').length;

      console.log(`[Cand ${i}] area=${Math.abs(area/2).toFixed(0)} T=${topE.darkFrac.toFixed(2)} B=${bottomE.darkFrac.toFixed(2)} L=${leftE.darkFrac.toFixed(2)} R=${rightE.darkFrac.toFixed(2)} solid=${solidCount} dashed=${dashedCount}`);

      let rotation: Rotation = 0;
      let valid = false;

      // Robust corner analysis: count dark pixels to detect anomalies like Red 'X'
      const cornerPatch = (rFrac: number, cFrac: number) => {
        const r0 = Math.floor(rFrac * WARP_SIZE);
        const c0 = Math.floor(cFrac * WARP_SIZE);
        const SZ = 16;
        let sum = 0;
        let darkPxCount = 0;
        for (let r = r0; r < r0+SZ; r++) {
          for (let c = c0; c < c0+SZ; c++) {
             const v = px(c, r);
             sum += v;
             if (v < 140) darkPxCount++;
          }
        }
        return {avg: sum / (SZ*SZ), dark: darkPxCount};
      };

      // Extract 4 inner corner patches safely inside the border
      const tl = cornerPatch(0.12, 0.12);
      const tr = cornerPatch(0.12, 0.72);
      const br = cornerPatch(0.72, 0.72);
      const bl = cornerPatch(0.72, 0.12);

      const isOrient = (p: any) => p.dark > 100;
      const isEmpty  = (p: any) => p.dark < 15 && p.avg > 145;

      console.log(`[Cand ${i}] Corners: TL(d:${tl.dark},a:${tl.avg.toFixed(0)}) TR(d:${tr.dark},a:${tr.avg.toFixed(0)}) BR(d:${br.dark},a:${br.avg.toFixed(0)}) BL(d:${bl.dark},a:${bl.avg.toFixed(0)})`);

      // ── Marker 1: 4 solid edges, exactly 1 orientation square, exactly 3 empty corners ──
      if (solidCount === 4) {
        if      (isOrient(tl) && isEmpty(tr) && isEmpty(br) && isEmpty(bl)) { rotation = 0; valid = true; }
        else if (isOrient(tr) && isEmpty(br) && isEmpty(bl) && isEmpty(tl)) { rotation = 270; valid = true; }
        else if (isOrient(br) && isEmpty(bl) && isEmpty(tl) && isEmpty(tr)) { rotation = 180; valid = true; }
        else if (isOrient(bl) && isEmpty(tl) && isEmpty(tr) && isEmpty(br)) { rotation = 90; valid = true; }
      } 
      // ── Marker 2: 2 solid edges, 2 dashed edges, exactly 4 empty corners ──
      else if (solidCount === 2 && dashedCount === 2) {
        if (isEmpty(tl) && isEmpty(tr) && isEmpty(br) && isEmpty(bl)) {
          if      (leftType === 'SOLID'  && bottomType === 'SOLID') { rotation = 0;   valid = true; }
          else if (topType === 'SOLID'   && leftType === 'SOLID')   { rotation = 270; valid = true; }
          else if (topType === 'SOLID'   && rightType === 'SOLID')  { rotation = 180; valid = true; }
          else if (rightType === 'SOLID' && bottomType === 'SOLID') { rotation = 90;  valid = true; }
        }
      }

      if (!valid) continue;

      console.log(`[Detection] ✅ Marker accepted, rotation=${rotation}°`);

      // Re-warp to full 300×300 from original raw image
      const absCorners = cand.map((p: any) => ({x: p.x + sX, y: p.y + sY}));
      const OUT = 300;
      const m = aruco.CV.getPerspectiveTransform(absCorners, OUT - 1);
      let finalBuf = warpRGBA(rawData, W, H, OUT, OUT, m);
      finalBuf = rotateBuffer(finalBuf, OUT, OUT, rotation).data;
      return {data: finalBuf, width: OUT, height: OUT, rotation};
    }

    console.log('[Detection] No valid marker found.');
    return null;
  };

  // ── Main detect function ──────────────────────────────────────────────────
  const detectMarker = async (imagePath: string, photoW: number, photoH: number) => {
    const t0 = Date.now();
    try {
      // Resize to 800×800 for faster processing (keep square aspect for symmetric scan)
      const PROC_SIZE = 800;
      const resized = await ImageResizer.createResizedImage(
        imagePath, PROC_SIZE, PROC_SIZE, 'JPEG', 90, 0, undefined, false, {mode: 'contain'},
      );
      console.log(`[Pipeline] Resized in ${Date.now()-t0}ms → ${resized.width}x${resized.height}`);

      const b64 = await RNFS.readFile(resized.uri, 'base64');
      const raw = jpeg.decode(Buffer.from(b64, 'base64'), {useTArray: true});
      console.log(`[Pipeline] Decoded ${raw.width}x${raw.height} in ${Date.now()-t0}ms`);

      const result = runDetection(raw.data as Uint8Array, raw.width, raw.height);
      if (!result) {
        setStatusMsg('❌ No Marker Found — try again');
        setDetected(false);
        setIsProcessing(false);
        return;
      }

      const {data: finalBuf, width: OUT_W, height: OUT_H, rotation} = result;
      const encoded = jpeg.encode({data: finalBuf, width: OUT_W, height: OUT_H}, 92);
      const b64out = Buffer.from(encoded.data).toString('base64');
      const finalPath = `${RNFS.CachesDirectoryPath}/marker_${Date.now()}.jpg`;
      await RNFS.writeFile(finalPath, b64out, 'base64');

      const elapsed = Date.now() - t0;
      console.log(`[Pipeline] ✅ Done ${elapsed}ms, rot=${rotation}°`);

      setCroppedImagePath('file://' + finalPath);
      setDetected(true);
      setStatusMsg(`✅ Marker detected! (${elapsed}ms | ${rotation}°)`);
      setMarkerList(prev => [{uri: 'file://' + finalPath, rot: rotation}, ...prev].slice(0, MAX_MARKERS));
    } catch (err: any) {
      const msg = err?.message ?? String(err);
      console.log('[Pipeline] Error:', msg);
      setStatusMsg(`⚠️ ${msg.substring(0, 100)}`);
      setDetected(false);
    } finally {
      setIsProcessing(false);
    }
  };

  // ── Capture ───────────────────────────────────────────────────────────────
  const takePhoto = async () => {
    if (!cameraRef.current) return;
    setIsProcessing(true);
    setDetected(false);
    setCroppedImagePath(null);
    setStatusMsg('📷 Capturing...');
    try {
      const photo = await cameraRef.current.takePhoto({
        flash: 'off',
        qualityPrioritization: 'quality',
      });
      console.log(`[Capture] ${photo.width}x${photo.height}`);
      await detectMarker('file://' + photo.path, photo.width, photo.height);
    } catch (e) {
      console.log('[Capture] Error:', e);
      setIsProcessing(false);
      setStatusMsg('⚠️ Capture failed');
    }
  };

  // ── Render ────────────────────────────────────────────────────────────────
  if (!device) {
    return (
      <View style={styles.center}>
        <Text style={styles.white}>Loading camera...</Text>
      </View>
    );
  }

  const hasGallery = markerList.length > 0;

  return (
    <View style={styles.root}>
      {/* Live camera feed */}
      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        format={format}
        isActive={true}
        photo={true}
      />

      {/* Dark overlays */}
      <View style={[styles.overlay, {top: 0, left: 0, right: 0, height: BOX_TOP}]} />
      <View style={[styles.overlay, {top: BOX_TOP+BOX_H, left: 0, right: 0, bottom: 0}]} />
      <View style={[styles.overlay, {top: BOX_TOP, left: 0, width: BOX_LEFT, height: BOX_H}]} />
      <View style={[styles.overlay, {top: BOX_TOP, left: BOX_LEFT+BOX_W, right: 0, height: BOX_H}]} />

      {/* Scan box */}
      <View style={[styles.scanBox, {top: BOX_TOP, left: BOX_LEFT, width: BOX_W, height: BOX_H}]}>
        <View style={[styles.corner, styles.cornerTL]} />
        <View style={[styles.corner, styles.cornerTR]} />
        <View style={[styles.corner, styles.cornerBL]} />
        <View style={[styles.corner, styles.cornerBR]} />
      </View>

      <Text style={[styles.guide, {top: BOX_TOP + 12}]}>Align marker inside box</Text>

      <Text style={[styles.statusText, {bottom: hasGallery ? 225 : 120}]}>
        {statusMsg}
      </Text>

      {/* Latest marker preview */}
      {croppedImagePath && (
        <View style={styles.previewCard}>
          <Text style={styles.previewLabel}>Extracted Marker (300×300)</Text>
          <Image source={{uri: croppedImagePath}} style={styles.previewImg} />
        </View>
      )}

      {/* Capture button */}
      <TouchableOpacity
        id="captureButton"
        style={[styles.captureBtn, {bottom: hasGallery ? 150 : 50}]}
        onPress={takePhoto}
        disabled={isProcessing}
        activeOpacity={0.8}>
        {isProcessing ? (
          <ActivityIndicator color="#00ff88" size="large" />
        ) : (
          <Text style={styles.captureText}>📷  Capture</Text>
        )}
      </TouchableOpacity>

      {/* Mini gallery for live progress */}
      {hasGallery && (
        <View style={styles.gallery}>
          <Text style={styles.galleryTitle}>
            Captured Markers — {markerList.length} / {MAX_MARKERS}
          </Text>
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={styles.galleryScroll}>
            {markerList.map((item, idx) => (
              <View key={idx} style={styles.thumb}>
                <Image source={{uri: item.uri}} style={styles.thumbImg} />
                <Text style={styles.thumbLabel}>#{markerList.length - idx}</Text>
                {item.rot !== 0 && (
                  <Text style={styles.thumbRot}>{item.rot}°</Text>
                )}
              </View>
            ))}
          </ScrollView>
        </View>
      )}

      {/* FINAL RESULTS UI - Spec 6: processed 20 markers displayed at the end on UI should all be exactly 300x300px */}
      <Modal visible={markerList.length >= MAX_MARKERS} animationType="slide">
        <View style={styles.resultsRoot}>
          <Text style={styles.resultsTitle}>All {MAX_MARKERS} Markers Captured!</Text>
          <ScrollView contentContainerStyle={styles.resultsScroll}>
            {markerList.map((item, idx) => (
              <View key={idx} style={styles.resultItem}>
                <Text style={styles.resultLabel}>Marker #{MAX_MARKERS - idx}</Text>
                {/* Displayed EXACTLY at 300x300px as required by spec */}
                <Image source={{uri: item.uri}} style={styles.resultImg} />
                <Text style={styles.resultRot}>Rotation applied: {item.rot}°</Text>
              </View>
            ))}
          </ScrollView>
          <View style={styles.resultsFooter}>
            <TouchableOpacity 
               style={styles.captureBtn} 
               onPress={() => {
                 setMarkerList([]);
                 setCroppedImagePath(null);
                 setStatusMsg('Ready for new scan');
               }}>
              <Text style={styles.captureText}>Restart Scanner</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </View>
  );
}

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
  corner: {
    position: 'absolute',
    width: 22,
    height: 22,
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
    top: Math.max(20, BOX_TOP - 145),
    alignSelf: 'center',
    backgroundColor: 'rgba(10,10,20,0.88)',
    borderWidth: 1,
    borderColor: '#00ff88',
    borderRadius: 8,
    padding: 6,
    alignItems: 'center',
  },
  previewLabel: {color: '#aaa', fontSize: 10, marginBottom: 4},
  // Preview shows at 100×100 inline (full 300×300 stored on disk)
  previewImg: {width: 100, height: 100, resizeMode: 'stretch'},

  captureBtn: {
    position: 'absolute',
    alignSelf: 'center',
    backgroundColor: 'rgba(0,10,5,0.92)',
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

  // Gallery: horizontal scroll strip at the bottom
  gallery: {
    position: 'absolute',
    bottom: 0,
    width: '100%',
    backgroundColor: 'rgba(5,5,15,0.93)',
    borderTopWidth: 1,
    borderTopColor: '#1a1a2e',
    paddingTop: 6,
    paddingBottom: 8,
    height: 140,
  },
  galleryTitle: {
    color: '#666',
    fontSize: 10,
    fontWeight: 'bold',
    paddingHorizontal: 12,
    marginBottom: 4,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  galleryScroll: {paddingHorizontal: 6},
  thumb: {
    alignItems: 'center',
    marginHorizontal: 3,
    width: 108,
  },
  thumbImg: {
    width: 100,
    height: 100,
    resizeMode: 'stretch',
    borderRadius: 4,
    borderWidth: 1,
    borderColor: '#333',
  },
  thumbLabel: {color: '#555', fontSize: 9, marginTop: 2},
  thumbRot:  {color: '#00ff88', fontSize: 8, fontWeight: 'bold'},

  // Final Results Screen Styles
  resultsRoot: {flex: 1, backgroundColor: '#0a0a10'},
  resultsTitle: {
    color: '#00ff88',
    fontSize: 20,
    fontWeight: 'bold',
    textAlign: 'center',
    marginTop: 60,
    marginBottom: 20,
  },
  resultsScroll: {paddingBottom: 100, alignItems: 'center'},
  resultItem: {
    marginBottom: 30,
    alignItems: 'center',
    backgroundColor: '#111',
    padding: 15,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#222',
  },
  resultLabel: {color: '#fff', fontSize: 16, fontWeight: 'bold', marginBottom: 10},
  resultImg: {
    width: 300,
    height: 300,
    resizeMode: 'stretch',
    borderWidth: 2,
    borderColor: '#00ff88',
  },
  resultRot: {color: '#aaa', fontSize: 13, marginTop: 10},
  resultsFooter: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    height: 90,
    backgroundColor: 'rgba(10,10,16,0.95)',
    justifyContent: 'center',
    borderTopWidth: 1,
    borderTopColor: '#222',
  },
});