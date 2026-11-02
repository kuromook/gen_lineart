"""追補202 の診断(点数ではない)。`gakusai3` L29 SKIN の乳輪の窓で、マスク・塗りの縁が線と何 px ずれているかを探す。

層全体のずれ(±1px 以内)は同じ日の探索で確認済み。ここは窓 400px 四方の局所だけ。
"""
import sys; sys.path.insert(0,'tools/colour')
import numpy as np, cv2, json
import m102_b_standard_v2 as m102, m119_mask_faces_and_lines as m, m116_mask_material_adopted4 as m116, m12_score_a0_nulls as m12
kinds=m102.load_kinds(); R=16; WIN=200
pg,_=m.load_gakusai3(kinds); H,W=pg['H'],pg['W']
p=[p for p in json.load(open(m116.CM,encoding='utf-8'))['pages'] if pg['pp']==p['source']][0]
files={int(l['layer_id']):l['file'] for l in p['layers'] if l.get('file')}
mm={int(r['layer_id']):r for r in pg['mm']}
def edge(v): return v&~cv2.erode(v.astype(np.uint8),m.K4,borderValue=1).astype(bool)
def alpha(l):
    im=cv2.imread(str(m116.D/files[l]),cv2.IMREAD_UNCHANGED); return np.ascontiguousarray(im[:H,:W,3]>=m12.ALPHA_DRAWN)
mask=m.vis_side(pg['mdir']/mm[29]['mask_file'],H,W)
srcs={'L29 マスク':mask}
for l,nm in [(38,'color clip'),(52,'deep clip'),(64,'deep 非clip'),(63,'high light 非clip'),(53,'high light clip'),(54,'high light clip')]:
    srcs[f'L{l} {nm}']=alpha(l)
targets={'線':pg['ink'],'L29 マスク':mask}
dts={k:cv2.distanceTransform((~v).astype(np.uint8),cv2.DIST_L2,5) for k,v in targets.items()}
for cx,cy in [(2915,4585),(4087,4728)]:
    y0,y1,x0,x1=cy-WIN,cy+WIN,cx-WIN,cx+WIN
    print(f'--- 窓 x{x0}-{x1} y{y0}-{y1}')
    for sn,sv in srcs.items():
        e=edge(sv)[y0:y1,x0:x1]; ey,ex=np.nonzero(e)
        if len(ey)<30: print(f'  {sn:<22} 縁 {len(ey)}(少ない)'); continue
        ey+=y0; ex+=x0
        out=[]
        for tn,d in dts.items():
            if tn==sn: continue
            base=(d[ey,ex]<=1).mean(); best=(0,0,base)
            for dy in range(-R,R+1):
                for dx in range(-R,R+1):
                    s=(d[np.clip(ey+dy,0,H-1),np.clip(ex+dx,0,W-1)]<=1).mean()
                    if s>best[2]: best=(dx,dy,s)
            out.append(f'→{tn}: 0で {base:.2f} / 最良 ({best[0]:+d},{best[1]:+d}) {best[2]:.2f}')
        print(f'  {sn:<22} 縁 {len(ey):>5}  '+'   '.join(out))
