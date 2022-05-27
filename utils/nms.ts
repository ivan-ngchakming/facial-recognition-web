import { BBox } from "../types";

/**
 * source: https://github.com/erceth/non-maximum-suppression/blob/master/nms.js
 */
const nms = (bboxes: number[][], overlapThresh: number) => {
  if (bboxes.length === 0) {
    return [] as BBox[];
  }

  const pick: BBox[] = [];
  
  let candidates = bboxes.map(box => { // TODO: replace with vectorization
    return {
      x1: box[0],
      y1: box[1],
      x2: box[2],
      y2: box[3],
      width: box[4],
      height: box[5],
      area: box[4] * box[5]
    }
  });

  candidates.sort((b1, b2) => {
    return b1.y2 - b2.y2;
  });

  while (candidates.length > 0) {
    let last = candidates[candidates.length - 1];
    pick.push(last);
    let suppress = [last];

    for (let i = 0; i < candidates.length - 1; i ++) {
      const box = candidates[i];
      const xx1 = Math.max(box.x1, last.x1)
      const yy1 = Math.max(box.y1, last.y1)
      const xx2 = Math.min(box.x2, last.x2);
      const yy2 = Math.min(box.y2, last.y2)
      const w = Math.max(0, xx2 - xx1 + 1);
      const h = Math.max(0, yy2 - yy1 + 1);
      const overlap = (w * h ) / box.area;
      if (overlap > overlapThresh) {
        suppress.push(candidates[i])
      }
    }
    
    candidates = candidates.filter((box) => {
      return !suppress.find((supp) => {
        return supp === box;
      })
    });
  }
  return pick;
};

export default nms;
