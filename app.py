import os, io
import streamlit as st
import requests
import pandas as pd
from openpyxl import Workbook
from photo_engine import load_checklist, load_mapping, load_raw_data, header_extract, detect_candidates, build_report

BASE=os.path.dirname(__file__)
CHECKLIST=os.path.join(BASE,'master','Observation_Checklist.xlsx')
DASH=os.path.join(BASE,'master','PDI_Dashboard_QC.xlsx')
TRACEABILITY_URL="https://script.google.com/macros/s/AKfycbyy9M8-f9_Hy2vUp8xaFmE8oj1d0kwaR58mzf7S7W305hgU1xMPfyz0s6-xfnAB0qOUUQ/exec"

st.set_page_config(page_title='PDI Photo Upload – Phase 2', page_icon='📋', layout='centered')
st.title('📋 PDI Photo Upload – Phase 2')
st.caption('Paper checklist → Photo Upload → NOT OK detection → Traceability Report')

@st.cache_data

def load_data():
    checklist=load_checklist(CHECKLIST)
    mapping, conflicts=load_mapping(DASH) if os.path.exists(DASH) else ({}, {})
    raw=load_raw_data(DASH) if os.path.exists(DASH) else {}
    return checklist,mapping,conflicts,raw

checklist,mapping,conflicts,raw=load_data()

with st.form('upload'):
    inspector=st.text_input('Inspector Name *', placeholder='e.g. Harish')
    jpc_manual=st.text_input('JPC Number (optional – only if OCR cannot read it)', placeholder='e.g. VS-3009')
    photos=st.file_uploader('Upload checklist photos (all pages)', type=['jpg','jpeg','png'], accept_multiple_files=True)
    submitted=st.form_submit_button('Process Checklist')

if submitted:
    if not inspector.strip(): st.error('Inspector Name is required.'); st.stop()
    if not photos: st.error('Please upload at least one checklist photo.'); st.stop()
    all_candidates=[]; headers=[]
    for f in photos:
        data=f.read()
        import cv2, numpy as np
        img=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
        if img is None: continue
        headers.append(header_extract(img))
        c=detect_candidates(img,checklist)
        if not c.empty: all_candidates.append(c)
    jpcs=[h['jpc'] for h in headers if h.get('jpc')]
    jpc=(jpcs[0] if jpcs else jpc_manual.strip().upper())
    if not jpc:
        st.warning('JPC Number could not be read automatically. Enter it above and re-submit.')
        st.stop()
    candidates=pd.concat(all_candidates,ignore_index=True) if all_candidates else pd.DataFrame(columns=['Observation','OCR Match %','OK Ink','NOT OK Ink','Detected State'])
    if candidates.empty:
        st.success('No NOT OK candidates detected. If every point is genuinely OK, no Observation Log row is created.')
        st.stop()
    # Deduplicate repeated OCR hits from overlapping page areas
    candidates=candidates.sort_values(['Observation','Detected State']).drop_duplicates('Observation',keep='first').reset_index(drop=True)
    st.subheader('AI/CV detection review')
    st.dataframe(candidates,use_container_width=True,hide_index=True)
    st.info('REVIEW items are low-confidence and should be checked before export. This protects the live dashboard from false defects.')
    report,review=build_report(jpc,inspector,candidates,raw,mapping)
    # Send Traceability Report to Google Sheet
    bio=io.BytesIO(); wb.save(bio); bio.seek(0)
    st.download_button('⬇️ Download Traceability Report Excel',bio,'PDI_Automated_Traceability_Report_Phase2.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    st.caption(f'Inspector: {inspector} | JPC: {jpc} | Photos: {len(photos)} | Candidates: {len(report)}')
