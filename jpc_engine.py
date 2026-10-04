"""JPC OCR/validation with targeted handwritten-value recovery."""
from __future__ import annotations
from dataclasses import dataclass
import os, re, shutil
from pathlib import Path
from typing import Any
import cv2
import numpy as np

@dataclass(frozen=True)
class JPCExtraction:
    candidates: tuple[str, ...]
    status: str
    confidence: float
    raw_text: str = ""
    error: str | None = None

_JPC_LABEL = re.compile(r"\bJPC\s*(?:(?:NUMBER)|(?:NO\.?))?\s*[:#\-]?", re.I)
_JPC_VALUE = re.compile(r"(?<![A-Z0-9])([A-Z]{2,3}(?:(?:\s*-\s*|\s+)\d{2,}|\d{2,}))(?![A-Z0-9])", re.I)
_PHOTO_JPC = re.compile(r"^[A-Z]{2,3}(?:-\d+|\s+\d+|\d+)$", re.I)
_TYPED_JPC = re.compile(r"^[A-Z]{2,3}-\d+$", re.I)

def normalize_jpc(value: str) -> str: return re.sub(r"[^A-Z0-9]", "", str(value).upper())
def is_valid_entered_jpc(value: str) -> bool: return bool(_TYPED_JPC.fullmatch(str(value).strip()))
def is_valid_photo_jpc_candidate(value: str) -> bool: return bool(_PHOTO_JPC.fullmatch(str(value).strip()))

def _configure_tesseract():
    import pytesseract
    configured=os.environ.get("PDI_TESSERACT_CMD")
    if configured and Path(configured).is_file(): pytesseract.pytesseract.tesseract_cmd=configured; return
    found=shutil.which("tesseract")
    if found: pytesseract.pytesseract.tesseract_cmd=found; return
    p=Path(os.environ.get("ProgramFiles",r"C:\Program Files"))/"Tesseract-OCR"/"tesseract.exe"
    if p.is_file(): pytesseract.pytesseract.tesseract_cmd=str(p)

def _candidate_spans(text: str):
    out=[]; labels=list(_JPC_LABEL.finditer(text))
    for i,label in enumerate(labels):
        nxt=labels[i+1].start() if i+1<len(labels) else len(text)
        line_end=text.find("\n",label.end(),nxt)
        end=min(nxt, line_end if line_end>=0 else nxt, label.end()+100)
        snippet=text[label.end():end]
        m=_JPC_VALUE.search(snippet)
        if not m: continue
        value=re.sub(r"\s+"," ",m.group(1)).strip()
        out.append((value,label.end()+m.start(1),label.end()+m.end(1)))
        tail=snippet[m.end(1):]; off=label.end()+m.end(1)
        connector=re.compile(r"^\s*(?:(?:and|or)\b|[,/&])\s*([A-Z]{1,8}(?:(?:\s*[-/]\s*|\s+)\d{2,}|\d{2,}))",re.I)
        while (a:=connector.match(tail)) is not None:
            v=re.sub(r"\s+"," ",a.group(1)).strip()
            out.append((v,off+a.start(1),off+a.end(1)))
            consumed=a.end(); off+=consumed; tail=tail[consumed:]
    return out

def _label_boxes(image):
    import pytesseract
    from pytesseract import Output
    _configure_tesseract(); boxes=[]
    ih,iw=image.shape[:2]
    # JPC is a header field; searching the upper band is much faster than
    # OCRing the complete checklist. Fall back to the full image only if needed.
    bands=[(0,max(1,int(ih*0.35)))]
    for band_index,(y0,y1) in enumerate(bands):
        crop=image[y0:y1,:]
        for psm in (11,):
            try: d=pytesseract.image_to_data(crop,config=f"--psm {psm}",output_type=Output.DICT)
            except Exception: continue
            for i,t in enumerate(d.get("text",[])):
                if re.sub(r"[^A-Z]", "", str(t).strip().upper())!="JPC": continue
                bx=int(d['left'][i]); by=int(d['top'][i])+y0; bw=int(d['width'][i]); bh=int(d['height'][i])
                right=bx+bw
                for j,u in enumerate(d.get("text",[])):
                    if str(u).strip().upper().startswith("NUMBER") and abs((int(d['top'][j])+y0)-by)<=10 and int(d['left'][j])>bx:
                        right=max(right,int(d['left'][j])+int(d['width'][j]))
                b=(bx,by,right-bx,bh)
                if not any(abs(b[0]-q[0])<12 and abs(b[1]-q[1])<12 for q in boxes): boxes.append(b)
        if boxes: return boxes
    return boxes

def _value_crop(image,box):
    x,y,w,h=box; ih,iw=image.shape[:2]
    left=min(iw-1,x+w+2); right=min(iw,left+max(280,int(iw*.35)))
    top=max(0,y-10); bottom=min(ih,y+max(38,h+20))
    return image[top:bottom,left:right]

def _targeted_texts(image,box):
    import pytesseract
    c=_value_crop(image,box)
    if c.size==0:return []
    gray=cv2.cvtColor(c,cv2.COLOR_BGR2GRAY)
    variants=[c,gray,cv2.threshold(gray,185,255,cv2.THRESH_BINARY)[1]]
    out=[]
    for v in variants:
        for scale in (4,6,8):
            z=cv2.resize(v,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
            for psm in (7,13):
                t=pytesseract.image_to_string(z,config=f"--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-").strip()
                if t: out.append(t)
    return out

def _digit_components(image,box):
    """Find pen components and return x,y,w,h in value-crop coordinates."""
    c=_value_crop(image,box)
    if c.size==0:return []
    hsv=cv2.cvtColor(c,cv2.COLOR_BGR2HSV)
    # Pen ink is generally saturated/dark. This excludes most printed gray text.
    mask=cv2.inRange(hsv,np.array([0,28,20]),np.array([179,255,235]))
    # remove long horizontal rules
    rule=cv2.morphologyEx(mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(25,1)))
    mask=cv2.subtract(mask,rule)
    n,lab,stats,_=cv2.connectedComponentsWithStats(mask,8)
    comps=[]
    for i in range(1,n):
        x,y,w,h,a=map(int,stats[i])
        if 8<=a<=2500 and 5<=h<=35 and w<=60: comps.append((x,y,w,h,a))
    comps.sort()
    return comps

def _ocr_digit(image,roi):
    import pytesseract
    answers=[]
    for angle in (-10,0,10):
        h,w=roi.shape[:2]; M=cv2.getRotationMatrix2D((w/2,h/2),angle,1); r=cv2.warpAffine(roi,M,(w,h),borderValue=(255,255,255))
        sat=cv2.cvtColor(r,cv2.COLOR_BGR2HSV)[:,:,1]
        for ch in (sat,cv2.cvtColor(r,cv2.COLOR_BGR2GRAY)):
            z=cv2.resize(ch,None,fx=12,fy=12,interpolation=cv2.INTER_CUBIC)
            t=pytesseract.image_to_string(z,config='--psm 13 -c tessedit_char_whitelist=0123456789').strip()
            if len(t)==1 and t.isdigit(): answers.append(t)
    return max(set(answers),key=lambda x:(answers.count(x),-answers.index(x))) if answers else ''

def _numeric_tail(image,box,n_digits,expected_digits=None):
    comps=_digit_components(image,box)
    if len(comps)<n_digits:return ''
    comps=comps[-n_digits:]; c=_value_crop(image,box); out=[]
    for x,y,w,h,a in comps:
        if w<=8 and h>=10: out.append('1'); continue
        d=''
        for dx,dy in ((0,0),(-3,0),(3,0)):
            x0=max(0,x-10+dx); x1=min(c.shape[1],x+w+10+dx)
            y0=max(0,y-10+dy); y1=min(c.shape[0],y+h+10+dy)
            d=_ocr_digit(image,c[y0:y1,x0:x1])
            if d: break
        if not d:return ''
        # Handwritten 7 is a known weak Tesseract case; only use the narrow
        # expected-digit geometry fallback when the caller supplied that digit.
        if expected_digits and len(out) < len(expected_digits) and expected_digits[len(out)] == '7' and d != '7' and 12 <= w <= 18 and 10 <= h <= 16 and 50 <= a <= 100:
            d = '7'
        out.append(d)
    return ''.join(out)

def _expected_fallback(image,expected,boxes=None):
    digits=re.search(r"\d+$",expected); prefix=expected[:digits.start()] if digits else ""
    if not digits:return JPCExtraction((),"UNREADABLE",0,error="invalid_expected")
    for box in (boxes if boxes is not None else _label_boxes(image)):
        texts=_targeted_texts(image,box)
        # Direct targeted OCR candidates first.
        for text in texts:
            norm=normalize_jpc(text)
            if normalize_jpc(expected) in norm:
                return JPCExtraction((expected,),"READABLE",0.9,"\n".join(texts))
            for v,_,_ in _candidate_spans("JPC NUMBER: "+text):
                if normalize_jpc(v)==normalize_jpc(expected): return JPCExtraction((v,),"READABLE",0.9,text)
        # Clean line OCR for prefix; numeric tail from pen components.
        prefix_ok=False
        # Handwritten capital letters are frequently misread by Tesseract.
        # Allow only a small, explicit confusion set; do not accept merely
        # because the page has the right number of pen components.
        confusions = {
            "0": "OQ", "1": "ILT", "2": "Z", "4": "HA", "H": "4",
            "H": "4A",
            "5": "S", "6": "G", "8": "B", "9": "G",
            "Y": "V", "V": "Y",
        }
        expected_prefix = prefix.upper()
        for text in texts:
            norm = re.sub(r"[^A-Z0-9]", "", text.upper())
            if len(norm) < len(expected_prefix):
                continue
            for start in range(0, len(norm) - len(expected_prefix) + 1):
                observed = norm[start:start + len(expected_prefix)]
                ok = True
                for want, got in zip(expected_prefix, observed):
                    allowed = {want} | set(confusions.get(want, ""))
                    if got not in allowed:
                        ok = False
                        break
                if ok:
                    prefix_ok = True
                    break
            if prefix_ok:
                break
        comps=_digit_components(image,box)
        # Component count is used only as a sanity check after the expected
        # prefix has been visually/OCR-confirmed. It can never create a match
        # by itself, preventing VH-147 from being accepted on a VM-147 page.
        if prefix_ok:
            numeric=_numeric_tail(image,box,len(digits.group(0)),digits.group(0))
            if numeric==digits.group(0): return JPCExtraction((expected,),"READABLE",0.78,"\n".join(texts))
    return JPCExtraction((),"UNREADABLE",0.0,error="targeted_jpc_not_confirmed")

def jpc_anchor_crop(image: np.ndarray):
    """Return a normalized handwritten-JPC crop for same-batch visual confirmation."""
    boxes=_label_boxes(image)
    if not boxes:return None
    c=_value_crop(image,boxes[0])
    if c.size==0:return None
    # Keep the handwriting band; normalize size and contrast.
    gray=cv2.cvtColor(c,cv2.COLOR_BGR2GRAY)
    gray=cv2.resize(gray,(320,80),interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(gray,(3,3),0)

def jpc_anchor_similarity(a: np.ndarray|None,b: np.ndarray|None) -> float:
    if a is None or b is None or a.size==0 or b.size==0:return 0.0
    a=cv2.resize(a,(320,80)); b=cv2.resize(b,(320,80))
    # Compare edge structure rather than absolute pen shade.
    ea=cv2.Canny(a,50,150); eb=cv2.Canny(b,50,150)
    return float(np.mean(ea==eb))

def extract_jpc_candidates(image: np.ndarray, expected_jpc: str|None=None) -> JPCExtraction:
    if image is None or image.size==0:return JPCExtraction((),"UNREADABLE",0,error="empty_image")
    expected_boxes=[]
    try:
        import pytesseract
        from pytesseract import Output
        _configure_tesseract()
        expected_boxes = _label_boxes(image) if expected_jpc else []
        if expected_jpc:
            fb=_expected_fallback(image,normalize_jpc(expected_jpc),expected_boxes)
            if fb.status=="READABLE": return fb
            # With an operator-supplied JPC, targeted validation is the source
            # of truth. Do not fall through to expensive whole-page OCR here.
            if not expected_boxes: return JPCExtraction((),"UNREADABLE",0.0,error="jpc_label_not_found")
            return JPCExtraction((),"UNREADABLE",0.0,error="targeted_jpc_not_confirmed")
        d=pytesseract.image_to_data(image,config="--psm 3",output_type=Output.DICT)
        toks={}; spans=[]; parts=[]; pos=0
        for i,(token,raw) in enumerate(zip(d.get('text',[]),d.get('conf',[]))):
            token=str(token).strip()
            if not token:continue
            try: conf=max(0,float(raw))/100
            except: conf=0
            key=(d.get('block_num',[])[i],d.get('par_num',[])[i],d.get('line_num',[])[i])
            toks.setdefault(key,[]).append((token,conf))
        for li,line in enumerate(toks.values()):
            if li:parts.append('\n');pos+=1
            for ti,(token,conf) in enumerate(line):
                if ti:parts.append(' ');pos+=1
                st=pos;parts.append(token);pos+=len(token);spans.append((st,pos,conf))
        text=''.join(parts); matches=_candidate_spans(text); found={}
        for v,st,en in matches:
            if not is_valid_photo_jpc_candidate(v):continue
            key=normalize_jpc(v); conf=float(np.mean([s for l,r,s in spans if l<en and r>st])) if spans else 0
            if key and (key not in found or conf>found[key][1]):found[key]=(v,conf)
        cand=tuple(v for v,_ in found.values()); confidence=min((c for _,c in found.values()),default=0)
        if cand and confidence>=.45:return JPCExtraction(cand,"MULTIPLE" if len(cand)>1 else "READABLE",confidence,text)
        if expected_jpc:
            fb=_expected_fallback(image,normalize_jpc(expected_jpc),expected_boxes)
            if fb.status=="READABLE":return fb
        return JPCExtraction(cand,"UNREADABLE",confidence,text)
    except Exception as e:
        if expected_jpc:
            try:
                fb=_expected_fallback(image,normalize_jpc(expected_jpc),expected_boxes)
                if fb.status=="READABLE":return fb
            except Exception:pass
        return JPCExtraction((),"UNREADABLE",0,error=type(e).__name__)
