#!/usr/bin/env python3
from __future__ import annotations

import base64
import os
from pathlib import Path
import py_compile
import shutil
import signal
import subprocess
import sys
import time

HELPER = base64.b64decode('IyA9PT0gU1RBR0U2X1NCQVNfR1JJRF9CQVRDSF9WMiA9PT0KCmRlZiBfc3RhZ2U2X2dyaWRfYXZhaWxhYmxlX21lbW9yeV9ieXRlcygpIC0+IGludDoKICAgICIiIkJlc3QtZWZmb3J0IExpbnV4IGF2YWlsYWJsZS1tZW1vcnkgcXVlcnkgdXNlZCBvbmx5IGZvciBiYXRjaC1zaXplIHNhZmV0eS4iIiIKICAgIHRyeToKICAgICAgICBmb3IgbGluZSBpbiBQYXRoKCIvcHJvYy9tZW1pbmZvIikucmVhZF90ZXh0KGVuY29kaW5nPSJ1dGYtOCIpLnNwbGl0bGluZXMoKToKICAgICAgICAgICAgaWYgbGluZS5zdGFydHN3aXRoKCJNZW1BdmFpbGFibGU6Iik6CiAgICAgICAgICAgICAgICByZXR1cm4gaW50KGxpbmUuc3BsaXQoKVsxXSkgKiAxMDI0CiAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgIHBhc3MKICAgIHJldHVybiAwCgoKZGVmIF9zdGFnZTZfZ3JpZF93aW5kb3dzKAogICAgbl9pOiBpbnQsCiAgICBuX2o6IGludCwKICAgIG5fd2luOiBpbnQsCikgLT4gdHVwbGVbbnAubmRhcnJheSwgbnAubmRhcnJheV06CiAgICAiIiJQcmVwYXJlIGV4YWN0IFN0YU1QUyB3cmFwX2ZpbHRfZ2xvYmFsIHdpbmRvd3MuIiIiCiAgICBuX2luYyA9IG1heCgxLCBpbnQobl93aW4pIC8vIDIpCiAgICBuX3dpbl9pID0gbWF4KDEsIG1hdGguY2VpbChpbnQobl9pKSAvIG5faW5jKSAtIDEpCiAgICBuX3dpbl9qID0gbWF4KDEsIG1hdGguY2VpbChpbnQobl9qKSAvIG5faW5jKSAtIDEpCgogICAgaGFsZiA9IGludChuX3dpbikgLy8gMgogICAgeCA9IG5wLmFyYW5nZSgxLCBoYWxmICsgMSwgZHR5cGU9bnAuZmxvYXQzMikKICAgIFgsIFkgPSBucC5tZXNoZ3JpZCh4LCB4KQogICAgYmFzZSA9IG5wLmNvbmNhdGVuYXRlKChYICsgWSwgbnAuZmxpcGxyKFggKyBZKSksIGF4aXM9MSkKICAgIGJhc2UgPSBucC5jb25jYXRlbmF0ZSgoYmFzZSwgbnAuZmxpcHVkKGJhc2UpKSwgYXhpcz0wKS5hc3R5cGUobnAuZmxvYXQzMikKCiAgICB3aW5kb3dzID0gbnAuZW1wdHkoKG5fd2luX2kgKiBuX3dpbl9qLCA2KSwgZHR5cGU9bnAuaW50MzIpCiAgICBwb3NpdGlvbiA9IDAKICAgIGZvciBpeDEgaW4gcmFuZ2Uobl93aW5faSk6CiAgICAgICAgaTEgPSBpeDEgKiBuX2luYwogICAgICAgIGkyID0gaTEgKyBpbnQobl93aW4pCiAgICAgICAgaV9zaGlmdCA9IDAKICAgICAgICBpZiBpMiA+IG5faToKICAgICAgICAgICAgaV9zaGlmdCA9IGkyIC0gbl9pCiAgICAgICAgICAgIGkyID0gbl9pCiAgICAgICAgICAgIGkxID0gbl9pIC0gaW50KG5fd2luKQoKICAgICAgICBmb3IgaXgyIGluIHJhbmdlKG5fd2luX2opOgogICAgICAgICAgICBqMSA9IGl4MiAqIG5faW5jCiAgICAgICAgICAgIGoyID0gajEgKyBpbnQobl93aW4pCiAgICAgICAgICAgIGpfc2hpZnQgPSAwCiAgICAgICAgICAgIGlmIGoyID4gbl9qOgogICAgICAgICAgICAgICAgal9zaGlmdCA9IGoyIC0gbl9qCiAgICAgICAgICAgICAgICBqMiA9IG5fagogICAgICAgICAgICAgICAgajEgPSBuX2ogLSBpbnQobl93aW4pCgogICAgICAgICAgICB3aW5kb3dzW3Bvc2l0aW9uXSA9IChpMSwgaTIsIGoxLCBqMiwgaV9zaGlmdCwgal9zaGlmdCkKICAgICAgICAgICAgcG9zaXRpb24gKz0gMQoKICAgIHJldHVybiB3aW5kb3dzLCBiYXNlCgoKZGVmIF9zdGFnZTZfZ3JpZF9hY3RpdmVfd2luZG93cygKICAgIG9jY3VwYW5jeTogbnAubmRhcnJheSwKICAgIHdpbmRvd3M6IG5wLm5kYXJyYXksCikgLT4gbnAubmRhcnJheToKICAgIG9jY3VwaWVkID0gbnAuYXNhcnJheShvY2N1cGFuY3ksIGR0eXBlPWJvb2wpCiAgICBpbnRlZ3JhbCA9IG5wLnplcm9zKAogICAgICAgIChvY2N1cGllZC5zaGFwZVswXSArIDEsIG9jY3VwaWVkLnNoYXBlWzFdICsgMSksCiAgICAgICAgZHR5cGU9bnAuaW50NjQsCiAgICApCiAgICBpbnRlZ3JhbFsxOiwgMTpdID0gKAogICAgICAgIG9jY3VwaWVkLmFzdHlwZShucC5pbnQ2NCwgY29weT1GYWxzZSkKICAgICAgICAuY3Vtc3VtKGF4aXM9MCwgZHR5cGU9bnAuaW50NjQpCiAgICAgICAgLmN1bXN1bShheGlzPTEsIGR0eXBlPW5wLmludDY0KQogICAgKQoKICAgIGkxID0gd2luZG93c1s6LCAwXS5hc3R5cGUobnAuaW50NjQpCiAgICBpMiA9IHdpbmRvd3NbOiwgMV0uYXN0eXBlKG5wLmludDY0KQogICAgajEgPSB3aW5kb3dzWzosIDJdLmFzdHlwZShucC5pbnQ2NCkKICAgIGoyID0gd2luZG93c1s6LCAzXS5hc3R5cGUobnAuaW50NjQpCgogICAgY291bnRzID0gKAogICAgICAgIGludGVncmFsW2kyLCBqMl0KICAgICAgICAtIGludGVncmFsW2kxLCBqMl0KICAgICAgICAtIGludGVncmFsW2kyLCBqMV0KICAgICAgICArIGludGVncmFsW2kxLCBqMV0KICAgICkKICAgIHJldHVybiBucC5mbGF0bm9uemVybyhjb3VudHMgPiAwKS5hc3R5cGUobnAuaW50NjQpCgoKZGVmIF9zdGFnZTZfd2luZG93X3dlaWdodHMoCiAgICBiYXNlX3dlaWdodDogbnAubmRhcnJheSwKICAgIHdpbmRvd3M6IG5wLm5kYXJyYXksCiAgICBpbmRpY2VzOiBucC5uZGFycmF5LAopIC0+IG5wLm5kYXJyYXk6CiAgICBuX3dpbiA9IGludChiYXNlX3dlaWdodC5zaGFwZVswXSkKICAgIG91dHB1dCA9IG5wLmVtcHR5KChpbmRpY2VzLnNpemUsIG5fd2luLCBuX3dpbiksIGR0eXBlPW5wLmZsb2F0MzIpCgogICAgZm9yIGxvY2FsLCBwcmVwYXJlZF9pbmRleCBpbiBlbnVtZXJhdGUoaW5kaWNlcyk6CiAgICAgICAgaV9zaGlmdCA9IGludCh3aW5kb3dzW2ludChwcmVwYXJlZF9pbmRleCksIDRdKQogICAgICAgIGpfc2hpZnQgPSBpbnQod2luZG93c1tpbnQocHJlcGFyZWRfaW5kZXgpLCA1XSkKCiAgICAgICAgd2VpZ2h0ID0gYmFzZV93ZWlnaHQKICAgICAgICBpZiBpX3NoaWZ0ID4gMDoKICAgICAgICAgICAgc2hpZnRlZCA9IG5wLnplcm9zX2xpa2UoYmFzZV93ZWlnaHQpCiAgICAgICAgICAgIHNoaWZ0ZWRbaV9zaGlmdDosIDpdID0gYmFzZV93ZWlnaHRbOiBuX3dpbiAtIGlfc2hpZnQsIDpdCiAgICAgICAgICAgIHdlaWdodCA9IHNoaWZ0ZWQKCiAgICAgICAgaWYgal9zaGlmdCA+IDA6CiAgICAgICAgICAgIHNoaWZ0ZWQgPSBucC56ZXJvc19saWtlKGJhc2Vfd2VpZ2h0KQogICAgICAgICAgICBzaGlmdGVkWzosIGpfc2hpZnQ6XSA9IHdlaWdodFs6LCA6IG5fd2luIC0gal9zaGlmdF0KICAgICAgICAgICAgd2VpZ2h0ID0gc2hpZnRlZAoKICAgICAgICBvdXRwdXRbbG9jYWxdID0gd2VpZ2h0CgogICAgcmV0dXJuIG91dHB1dAoKCmRlZiBfc3RhZ2U2X2dvbGRzdGVpbl9maWx0ZXJfZGVuc2VfYmF0Y2goCiAgICBncmlkX3N0YWNrOiBucC5uZGFycmF5LAogICAgKiwKICAgIG5fd2luOiBpbnQsCiAgICBhbHBoYTogZmxvYXQsCiAgICBnb2xkX2ZsYWc6IGJvb2wsCiAgICBmZnRfd29ya2VyczogaW50LAogICAgd2luZG93X2JhdGNoOiBpbnQsCiAgICB3aW5kb3dzOiBucC5uZGFycmF5IHwgTm9uZSA9IE5vbmUsCiAgICBiYXNlX3dlaWdodDogbnAubmRhcnJheSB8IE5vbmUgPSBOb25lLAogICAgYWN0aXZlX2luZGljZXM6IG5wLm5kYXJyYXkgfCBOb25lID0gTm9uZSwKKSAtPiB0dXBsZVtucC5uZGFycmF5LCBucC5uZGFycmF5XToKICAgICIiIgogICAgVmVjdG9yaXplZCBlcXVpdmFsZW50IG9mIHBvcnRlZC5fd3JhcF9maWx0X2dsb2JhbCBmb3Igc2V2ZXJhbCBJRkdzLgoKICAgIFdpbmRvdyBvcmRlciwgb3ZlcmxhcC1hZGQgb3JkZXIsIHdlaWdodGluZywgR2F1c3NpYW4gc3BlY3RyYWwgc21vb3RoaW5nLAogICAgR29sZHN0ZWluIGV4cG9uZW50IGFuZCBsb3ctcGFzcyBrZXJuZWwgbWlycm9yIHRoZSBsZWdhY3kgaW1wbGVtZW50YXRpb24uCiAgICBFbXB0eSB3aW5kb3dzIGFyZSBza2lwcGVkIGJlY2F1c2UgdGhlaXIgZXhhY3QgY29udHJpYnV0aW9uIGlzIHplcm8uCiAgICAiIiIKICAgIHNvdXJjZSA9IG5wLmFzYXJyYXkoZ3JpZF9zdGFjaywgZHR5cGU9bnAuY29tcGxleDY0KQogICAgaWYgc291cmNlLm5kaW0gIT0gMzoKICAgICAgICByYWlzZSBTdGFnZTZTYmFzRXJyb3IoIkdSSUQgYmF0Y2ggaW5wdXQgbXVzdCBiZSBbcm93LCBjb2wsIGlmZ10iKQoKICAgIG5faSwgbl9qLCBuX2lmZ19iYXRjaCA9IHNvdXJjZS5zaGFwZQogICAgbl93aW4gPSBpbnQobl93aW4pCiAgICBuX3BhZCA9IGludChyb3VuZChuX3dpbiAqIDAuMjUpKQogICAgbl9leCA9IG5fd2luICsgbl9wYWQKCiAgICBpZiB3aW5kb3dzIGlzIE5vbmUgb3IgYmFzZV93ZWlnaHQgaXMgTm9uZToKICAgICAgICB3aW5kb3dzLCBiYXNlX3dlaWdodCA9IF9zdGFnZTZfZ3JpZF93aW5kb3dzKG5faSwgbl9qLCBuX3dpbikKICAgIGlmIGFjdGl2ZV9pbmRpY2VzIGlzIE5vbmU6CiAgICAgICAgYWN0aXZlX2luZGljZXMgPSBfc3RhZ2U2X2dyaWRfYWN0aXZlX3dpbmRvd3MoCiAgICAgICAgICAgIG5wLmFueShzb3VyY2UgIT0gMCwgYXhpcz0yKSwKICAgICAgICAgICAgd2luZG93cywKICAgICAgICApCgogICAgZmlsdGVyZWRfb3V0ID0gbnAuemVyb3Moc291cmNlLnNoYXBlLCBkdHlwZT1ucC5jb21wbGV4NjQsIG9yZGVyPSJGIikKICAgIGxvd3Bhc3Nfb3V0ID0gbnAuemVyb3Moc291cmNlLnNoYXBlLCBkdHlwZT1ucC5jb21wbGV4NjQsIG9yZGVyPSJGIikKICAgIGlmIGFjdGl2ZV9pbmRpY2VzLnNpemUgPT0gMDoKICAgICAgICByZXR1cm4gZmlsdGVyZWRfb3V0LCBsb3dwYXNzX291dAoKICAgIGltcG9ydCBweXN0YW1wcy5waXBlbGluZS5wb3J0ZWQgYXMgX3BvcnRlZAoKICAgIGdhdXNzaWFuID0gbnAuYXNhcnJheShfcG9ydGVkLl9nYXVzc3dpbig3KSwgZHR5cGU9bnAuZmxvYXQ2NCkKICAgIGcxNiA9IG5wLmFzYXJyYXkoX3BvcnRlZC5fZ2F1c3N3aW4obl9leCwgYWxwaGE9MTYuMCksIGR0eXBlPW5wLmZsb2F0NjQpCiAgICBsb3dfa2VybmVsID0gbnAuZmZ0LmlmZnRzaGlmdChucC5vdXRlcihnMTYsIGcxNikpLmFzdHlwZShucC5mbG9hdDY0KQoKICAgIHdpbmRvd19iYXRjaCA9IG1heCgxLCBtaW4oaW50KHdpbmRvd19iYXRjaCksIGludChhY3RpdmVfaW5kaWNlcy5zaXplKSkpCiAgICBmZnRfd29ya2VycyA9IG1heCgxLCBpbnQoZmZ0X3dvcmtlcnMpKQoKICAgIGZvciBiYXRjaF9zdGFydCBpbiByYW5nZSgwLCBpbnQoYWN0aXZlX2luZGljZXMuc2l6ZSksIHdpbmRvd19iYXRjaCk6CiAgICAgICAgYmF0Y2hfaW5kaWNlcyA9IGFjdGl2ZV9pbmRpY2VzW2JhdGNoX3N0YXJ0IDogYmF0Y2hfc3RhcnQgKyB3aW5kb3dfYmF0Y2hdCiAgICAgICAgY3VycmVudCA9IGludChiYXRjaF9pbmRpY2VzLnNpemUpCgogICAgICAgIHBoYXNlID0gbnAuemVyb3MoCiAgICAgICAgICAgIChjdXJyZW50LCBuX2lmZ19iYXRjaCwgbl9leCwgbl9leCksCiAgICAgICAgICAgIGR0eXBlPW5wLmNvbXBsZXgxMjgsCiAgICAgICAgKQogICAgICAgIGZvciBsb2NhbCwgcHJlcGFyZWRfaW5kZXggaW4gZW51bWVyYXRlKGJhdGNoX2luZGljZXMpOgogICAgICAgICAgICBpMSwgaTIsIGoxLCBqMiA9ICgKICAgICAgICAgICAgICAgIGludCh2YWx1ZSkKICAgICAgICAgICAgICAgIGZvciB2YWx1ZSBpbiB3aW5kb3dzW2ludChwcmVwYXJlZF9pbmRleCksIDo0XQogICAgICAgICAgICApCiAgICAgICAgICAgIHBoYXNlW2xvY2FsLCA6LCA6bl93aW4sIDpuX3dpbl0gPSBucC5tb3ZlYXhpcygKICAgICAgICAgICAgICAgIHNvdXJjZVtpMTppMiwgajE6ajIsIDpdLAogICAgICAgICAgICAgICAgMiwKICAgICAgICAgICAgICAgIDAsCiAgICAgICAgICAgICkKCiAgICAgICAgcGhhc2VfZmZ0ID0gc2NpcHlfZmZ0LmZmdDIoCiAgICAgICAgICAgIHBoYXNlLAogICAgICAgICAgICBheGVzPSgtMiwgLTEpLAogICAgICAgICAgICB3b3JrZXJzPWZmdF93b3JrZXJzLAogICAgICAgICkKICAgICAgICBhbXBsaXR1ZGUgPSBucC5hYnMocGhhc2VfZmZ0KQogICAgICAgIHNoaWZ0ZWQgPSBzY2lweV9mZnQuZmZ0c2hpZnQoYW1wbGl0dWRlLCBheGVzPSgtMiwgLTEpKQogICAgICAgIHNtb290aF9maXJzdCA9IG5kaW1hZ2UuY29udm9sdmUxZCgKICAgICAgICAgICAgc2hpZnRlZCwKICAgICAgICAgICAgZ2F1c3NpYW4sCiAgICAgICAgICAgIGF4aXM9LTIsCiAgICAgICAgICAgIG1vZGU9ImNvbnN0YW50IiwKICAgICAgICAgICAgY3ZhbD0wLjAsCiAgICAgICAgKQogICAgICAgIHNtb290aF9zZWNvbmQgPSBuZGltYWdlLmNvbnZvbHZlMWQoCiAgICAgICAgICAgIHNtb290aF9maXJzdCwKICAgICAgICAgICAgZ2F1c3NpYW4sCiAgICAgICAgICAgIGF4aXM9LTEsCiAgICAgICAgICAgIG1vZGU9ImNvbnN0YW50IiwKICAgICAgICAgICAgY3ZhbD0wLjAsCiAgICAgICAgKQogICAgICAgIHNwZWN0cnVtID0gc2NpcHlfZmZ0LmlmZnRzaGlmdChzbW9vdGhfc2Vjb25kLCBheGVzPSgtMiwgLTEpKQogICAgICAgIG1lZGlhbiA9IG5wLm1lZGlhbigKICAgICAgICAgICAgc3BlY3RydW0sCiAgICAgICAgICAgIGF4aXM9KC0yLCAtMSksCiAgICAgICAgICAgIGtlZXBkaW1zPVRydWUsCiAgICAgICAgKQogICAgICAgIG5wLmRpdmlkZSgKICAgICAgICAgICAgc3BlY3RydW0sCiAgICAgICAgICAgIG1lZGlhbiwKICAgICAgICAgICAgb3V0PXNwZWN0cnVtLAogICAgICAgICAgICB3aGVyZT1tZWRpYW4gIT0gMCwKICAgICAgICApCiAgICAgICAgbnAucG93ZXIoc3BlY3RydW0sIGZsb2F0KGFscGhhKSwgb3V0PXNwZWN0cnVtKQoKICAgICAgICBnb2xkc3RlaW4gPSBzY2lweV9mZnQuaWZmdDIoCiAgICAgICAgICAgIHBoYXNlX2ZmdCAqIHNwZWN0cnVtLAogICAgICAgICAgICBheGVzPSgtMiwgLTEpLAogICAgICAgICAgICB3b3JrZXJzPWZmdF93b3JrZXJzLAogICAgICAgICkKICAgICAgICBsb3dwYXNzID0gc2NpcHlfZmZ0LmlmZnQyKAogICAgICAgICAgICBwaGFzZV9mZnQgKiBsb3dfa2VybmVsW05vbmUsIE5vbmUsIDosIDpdLAogICAgICAgICAgICBheGVzPSgtMiwgLTEpLAogICAgICAgICAgICB3b3JrZXJzPWZmdF93b3JrZXJzLAogICAgICAgICkKCiAgICAgICAgd2VpZ2h0cyA9IF9zdGFnZTZfd2luZG93X3dlaWdodHMoCiAgICAgICAgICAgIGJhc2Vfd2VpZ2h0LAogICAgICAgICAgICB3aW5kb3dzLAogICAgICAgICAgICBiYXRjaF9pbmRpY2VzLAogICAgICAgICkKICAgICAgICBmb3IgbG9jYWwsIHByZXBhcmVkX2luZGV4IGluIGVudW1lcmF0ZShiYXRjaF9pbmRpY2VzKToKICAgICAgICAgICAgaTEsIGkyLCBqMSwgajIgPSAoCiAgICAgICAgICAgICAgICBpbnQodmFsdWUpCiAgICAgICAgICAgICAgICBmb3IgdmFsdWUgaW4gd2luZG93c1tpbnQocHJlcGFyZWRfaW5kZXgpLCA6NF0KICAgICAgICAgICAgKQogICAgICAgICAgICB3ZWlnaHQgPSB3ZWlnaHRzW2xvY2FsLCA6LCA6LCBOb25lXQoKICAgICAgICAgICAgaWYgZ29sZF9mbGFnOgogICAgICAgICAgICAgICAgY29udHJpYnV0aW9uID0gKAogICAgICAgICAgICAgICAgICAgIG5wLm1vdmVheGlzKAogICAgICAgICAgICAgICAgICAgICAgICBnb2xkc3RlaW5bbG9jYWwsIDosIDpuX3dpbiwgOm5fd2luXSwKICAgICAgICAgICAgICAgICAgICAgICAgMCwKICAgICAgICAgICAgICAgICAgICAgICAgMiwKICAgICAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAgICAgKiB3ZWlnaHQKICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIGZpbHRlcmVkX291dFtpMTppMiwgajE6ajIsIDpdID0gKAogICAgICAgICAgICAgICAgICAgIGZpbHRlcmVkX291dFtpMTppMiwgajE6ajIsIDpdCiAgICAgICAgICAgICAgICAgICAgKyBjb250cmlidXRpb24KICAgICAgICAgICAgICAgICkuYXN0eXBlKG5wLmNvbXBsZXg2NCkKCiAgICAgICAgICAgIGxvd19jb250cmlidXRpb24gPSAoCiAgICAgICAgICAgICAgICBucC5tb3ZlYXhpcygKICAgICAgICAgICAgICAgICAgICBsb3dwYXNzW2xvY2FsLCA6LCA6bl93aW4sIDpuX3dpbl0sCiAgICAgICAgICAgICAgICAgICAgMCwKICAgICAgICAgICAgICAgICAgICAyLAogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgKiB3ZWlnaHQKICAgICAgICAgICAgKQogICAgICAgICAgICBsb3dwYXNzX291dFtpMTppMiwgajE6ajIsIDpdID0gKAogICAgICAgICAgICAgICAgbG93cGFzc19vdXRbaTE6aTIsIGoxOmoyLCA6XQogICAgICAgICAgICAgICAgKyBsb3dfY29udHJpYnV0aW9uCiAgICAgICAgICAgICkuYXN0eXBlKG5wLmNvbXBsZXg2NCkKCiAgICBtYWduaXR1ZGUgPSBucC5hYnMoc291cmNlKS5hc3R5cGUobnAuZmxvYXQzMiwgY29weT1GYWxzZSkKICAgIGlmIGdvbGRfZmxhZzoKICAgICAgICBmaWx0ZXJlZF9vdXQgPSAoCiAgICAgICAgICAgIG1hZ25pdHVkZQogICAgICAgICAgICAqIG5wLmV4cCgxaiAqIG5wLmFuZ2xlKGZpbHRlcmVkX291dCkpCiAgICAgICAgKS5hc3R5cGUobnAuY29tcGxleDY0KQogICAgZWxzZToKICAgICAgICBmaWx0ZXJlZF9vdXQgPSBzb3VyY2UuY29weSgpCgogICAgbG93cGFzc19vdXQgPSAoCiAgICAgICAgbWFnbml0dWRlCiAgICAgICAgKiBucC5leHAoMWogKiBucC5hbmdsZShsb3dwYXNzX291dCkpCiAgICApLmFzdHlwZShucC5jb21wbGV4NjQpCiAgICByZXR1cm4gZmlsdGVyZWRfb3V0LCBsb3dwYXNzX291dAoKCmRlZiBfc3RhZ2U2X2F0b21pY19qc29uKHBhdGg6IFBhdGgsIHBheWxvYWQ6IGRpY3Rbc3RyLCBBbnldKSAtPiBOb25lOgogICAgdGVtcG9yYXJ5ID0gcGF0aC53aXRoX3N1ZmZpeChwYXRoLnN1ZmZpeCArICIudG1wIikKICAgIHRlbXBvcmFyeS53cml0ZV90ZXh0KAogICAgICAgIGpzb24uZHVtcHMocGF5bG9hZCwgaW5kZW50PTIsIGVuc3VyZV9hc2NpaT1GYWxzZSksCiAgICAgICAgZW5jb2Rpbmc9InV0Zi04IiwKICAgICkKICAgIHRlbXBvcmFyeS5yZXBsYWNlKHBhdGgpCgoKZGVmIF9zdGFnZTZfYnVpbGRfZ3JpZF9iYXRjaGVkKAogICAgKiwKICAgIHBoX2luOiBucC5uZGFycmF5LAogICAgbGluMDogbnAubmRhcnJheSwKICAgIG5faTogaW50LAogICAgbl9qOiBpbnQsCiAgICBuX3dpbjogaW50LAogICAgYWxwaGE6IGZsb2F0LAogICAgZ29sZF9mbGFnOiBib29sLAogICAgd29ya19kaXI6IFBhdGgsCiAgICBwcm9ncmVzczogYm9vbCwKKSAtPiB0dXBsZVsKICAgIG5wLm5kYXJyYXksCiAgICBucC5uZGFycmF5LAogICAgbnAubmRhcnJheSwKICAgIG5wLm5kYXJyYXksCiAgICBucC5uZGFycmF5LAogICAgaW50LAogICAgZGljdFtzdHIsIEFueV0sCl06CiAgICAiIiIKICAgIEJ1aWxkL2ZpbHRlciB0aGUgU0JBUyBncmlkIGluIHJlc3VtYWJsZSBJRkcgYmF0Y2hlcy4KCiAgICBGdWxsIFtncmlkX3JvdywgZ3JpZF9jb2wsIGFsbF9pZmddIGFycmF5cyBhcmUgbmV2ZXIgYWxsb2NhdGVkLiBGaWx0ZXJlZAogICAgZ3JpZC1wb2ludCBvdXRwdXRzIGFyZSBwZXJzaXN0ZW50IE5QWSBtZW1tYXBzLCBzbyBjb21wbGV0ZWQgSUZHcyBzdXJ2aXZlCiAgICBpbnRlcnJ1cHRpb24gYW5kIGNhbiBiZSByZXN1bWVkLgogICAgIiIiCiAgICBzb3VyY2UgPSBucC5hc2FycmF5KHBoX2luLCBkdHlwZT1ucC5jb21wbGV4NjQpCiAgICBuX3BzLCBuX2lmZyA9IHNvdXJjZS5zaGFwZQoKICAgIGlmZ19iYXRjaCA9IF9lbnZfaW50KCJQWVNUQU1QU19TVEFHRTZfR1JJRF9JRkdfQkFUQ0giLCA0KQogICAgd2luZG93X2JhdGNoID0gX2Vudl9pbnQoIlBZU1RBTVBTX1NUQUdFNl9HUklEX1dJTkRPV19CQVRDSCIsIDMyKQogICAgZmZ0X3dvcmtlcnMgPSBfZW52X2ludCgKICAgICAgICAiUFlTVEFNUFNfU1RBR0U2X0dSSURfRkZUX1dPUktFUlMiLAogICAgICAgIG1heCgxLCBtaW4oMTYsIG9zLmNwdV9jb3VudCgpIG9yIDEpKSwKICAgICkKICAgIHJlc3VtZSA9IF9lbnZfYm9vbCgiUFlTVEFNUFNfU1RBR0U2X0dSSURfUkVTVU1FIiwgVHJ1ZSkKCiAgICBhdmFpbGFibGUgPSBfc3RhZ2U2X2dyaWRfYXZhaWxhYmxlX21lbW9yeV9ieXRlcygpCiAgICBpZiBhdmFpbGFibGUgPiAwOgogICAgICAgIHBlcl9pZmdfYnl0ZXMgPSBpbnQobl9pKSAqIGludChuX2opICogOCAqIDMKICAgICAgICBzYWZlX2JhdGNoID0gbWF4KDEsIGludCgoYXZhaWxhYmxlICogMC4zNSkgLy8gbWF4KDEsIHBlcl9pZmdfYnl0ZXMpKSkKICAgICAgICBpZmdfYmF0Y2ggPSBtaW4oaWZnX2JhdGNoLCBzYWZlX2JhdGNoKQogICAgaWZnX2JhdGNoID0gbWF4KDEsIG1pbihpZmdfYmF0Y2gsIG5faWZnKSkKCiAgICBvcmRlciA9IG5wLmFyZ3NvcnQobnAuYXNhcnJheShsaW4wLCBkdHlwZT1ucC5pbnQ2NCksIGtpbmQ9InN0YWJsZSIpCiAgICBsaW5fc29ydGVkID0gbnAuYXNhcnJheShsaW4wLCBkdHlwZT1ucC5pbnQ2NClbb3JkZXJdCiAgICBzdGFydHMgPSBucC5jb25jYXRlbmF0ZSgKICAgICAgICAoCiAgICAgICAgICAgIG5wLmFzYXJyYXkoWzBdLCBkdHlwZT1ucC5pbnQ2NCksCiAgICAgICAgICAgIG5wLmZsYXRub256ZXJvKG5wLmRpZmYobGluX3NvcnRlZCkgIT0gMCkuYXN0eXBlKG5wLmludDY0KSArIDEsCiAgICAgICAgKQogICAgKQogICAgZ3JvdXBfbGluID0gbGluX3NvcnRlZFtzdGFydHNdCgogICAgZmlyc3RfdmFsdWVzID0gbnAuYWRkLnJlZHVjZWF0KAogICAgICAgIG5wLmFzYXJyYXkoc291cmNlW29yZGVyLCAwXSwgZHR5cGU9bnAuY29tcGxleDY0KSwKICAgICAgICBzdGFydHMsCiAgICAgICAgYXhpcz0wLAogICAgKQogICAgZmlyc3RfZmxhdCA9IG5wLnplcm9zKGludChuX2kpICogaW50KG5faiksIGR0eXBlPW5wLmNvbXBsZXg2NCkKICAgIGZpcnN0X2ZsYXRbZ3JvdXBfbGluXSA9IGZpcnN0X3ZhbHVlcwogICAgbnpfZmxhdCA9IGZpcnN0X2ZsYXQgIT0gMAogICAgaWYgbm90IG5wLmFueShuel9mbGF0KToKICAgICAgICByYWlzZSBTdGFnZTZTYmFzRXJyb3IoInV3X2dyaWQgaGFzIG5vIG5vbi16ZXJvIHBvaW50cyIpCgogICAgbnpfbGluID0gbnAuZmxhdG5vbnplcm8obnpfZmxhdCkuYXN0eXBlKG5wLmludDY0KQogICAgbnpfaSA9IG56X2xpbiAlIGludChuX2kpICsgMQogICAgbnpfaiA9IG56X2xpbiAvLyBpbnQobl9pKSArIDEKICAgIG5fZ3JpZF9wcyA9IGludChuel9saW4uc2l6ZSkKCiAgICBvY2N1cGFuY3kgPSBuel9mbGF0LnJlc2hhcGUoKGludChuX2kpLCBpbnQobl9qKSksIG9yZGVyPSJGIikKICAgIHdpbmRvd3MsIGJhc2Vfd2VpZ2h0ID0gX3N0YWdlNl9ncmlkX3dpbmRvd3MoCiAgICAgICAgaW50KG5faSksCiAgICAgICAgaW50KG5faiksCiAgICAgICAgaW50KG5fd2luKSwKICAgICkKICAgIGFjdGl2ZV9pbmRpY2VzID0gX3N0YWdlNl9ncmlkX2FjdGl2ZV93aW5kb3dzKAogICAgICAgIG9jY3VwYW5jeSwKICAgICAgICB3aW5kb3dzLAogICAgKQoKICAgIGRpZ2VzdCA9IGhhc2hsaWIuc2hhMjU2KCkKICAgIGRpZ2VzdC51cGRhdGUoCiAgICAgICAgbnAuYXNhcnJheSgKICAgICAgICAgICAgWwogICAgICAgICAgICAgICAgaW50KG5fcHMpLAogICAgICAgICAgICAgICAgaW50KG5faWZnKSwKICAgICAgICAgICAgICAgIGludChuX2kpLAogICAgICAgICAgICAgICAgaW50KG5faiksCiAgICAgICAgICAgICAgICBpbnQobl9ncmlkX3BzKSwKICAgICAgICAgICAgICAgIGludChuX3dpbiksCiAgICAgICAgICAgICAgICBpbnQoYm9vbChnb2xkX2ZsYWcpKSwKICAgICAgICAgICAgXSwKICAgICAgICAgICAgZHR5cGU9bnAuaW50NjQsCiAgICAgICAgKS50b2J5dGVzKCkKICAgICkKICAgIGRpZ2VzdC51cGRhdGUobnAuYXNhcnJheShbZmxvYXQoYWxwaGEpXSwgZHR5cGU9bnAuZmxvYXQ2NCkudG9ieXRlcygpKQogICAgZGlnZXN0LnVwZGF0ZShucC5hc2FycmF5KGxpbjAsIGR0eXBlPW5wLmludDY0KS50b2J5dGVzKCkpCiAgICBzaWduYXR1cmUgPSBkaWdlc3QuaGV4ZGlnZXN0KCkKCiAgICBncmlkX2RpciA9IFBhdGgod29ya19kaXIpIC8gImdyaWRfdjIiCiAgICBncmlkX2Rpci5ta2RpcihwYXJlbnRzPVRydWUsIGV4aXN0X29rPVRydWUpCiAgICBtZXRhX3BhdGggPSBncmlkX2RpciAvICJtZXRhLmpzb24iCiAgICBwaGFzZV9wYXRoID0gZ3JpZF9kaXIgLyAiZ3JpZF9waGFzZS5ucHkiCiAgICBsb3dfcGF0aCA9IGdyaWRfZGlyIC8gImdyaWRfbG93cGFzcy5ucHkiCiAgICBkb25lX3BhdGggPSBncmlkX2RpciAvICJkb25lLm5weSIKCiAgICBleGlzdGluZ19tZXRhOiBkaWN0W3N0ciwgQW55XSA9IHt9CiAgICBpZiBtZXRhX3BhdGguZXhpc3RzKCk6CiAgICAgICAgdHJ5OgogICAgICAgICAgICBleGlzdGluZ19tZXRhID0ganNvbi5sb2FkcyhtZXRhX3BhdGgucmVhZF90ZXh0KGVuY29kaW5nPSJ1dGYtOCIpKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIGV4aXN0aW5nX21ldGEgPSB7fQoKICAgIGNhbl9yZXN1bWUgPSAoCiAgICAgICAgcmVzdW1lCiAgICAgICAgYW5kIGV4aXN0aW5nX21ldGEuZ2V0KCJzaWduYXR1cmUiKSA9PSBzaWduYXR1cmUKICAgICAgICBhbmQgcGhhc2VfcGF0aC5leGlzdHMoKQogICAgICAgIGFuZCBsb3dfcGF0aC5leGlzdHMoKQogICAgICAgIGFuZCBkb25lX3BhdGguZXhpc3RzKCkKICAgICkKCiAgICBpZiBub3QgY2FuX3Jlc3VtZToKICAgICAgICBmb3IgcGF0aCBpbiAocGhhc2VfcGF0aCwgbG93X3BhdGgsIGRvbmVfcGF0aCwgbWV0YV9wYXRoKToKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgcGF0aC51bmxpbmsoKQogICAgICAgICAgICBleGNlcHQgRmlsZU5vdEZvdW5kRXJyb3I6CiAgICAgICAgICAgICAgICBwYXNzCgogICAgICAgIGdyaWRfcGhhc2UgPSBucC5saWIuZm9ybWF0Lm9wZW5fbWVtbWFwKAogICAgICAgICAgICBwaGFzZV9wYXRoLAogICAgICAgICAgICBtb2RlPSJ3KyIsCiAgICAgICAgICAgIGR0eXBlPW5wLmNvbXBsZXg2NCwKICAgICAgICAgICAgc2hhcGU9KG5fZ3JpZF9wcywgbl9pZmcpLAogICAgICAgICkKICAgICAgICBncmlkX2xvd3Bhc3MgPSBucC5saWIuZm9ybWF0Lm9wZW5fbWVtbWFwKAogICAgICAgICAgICBsb3dfcGF0aCwKICAgICAgICAgICAgbW9kZT0idysiLAogICAgICAgICAgICBkdHlwZT1ucC5jb21wbGV4NjQsCiAgICAgICAgICAgIHNoYXBlPShuX2dyaWRfcHMsIG5faWZnKSwKICAgICAgICApCiAgICAgICAgZG9uZSA9IG5wLmxpYi5mb3JtYXQub3Blbl9tZW1tYXAoCiAgICAgICAgICAgIGRvbmVfcGF0aCwKICAgICAgICAgICAgbW9kZT0idysiLAogICAgICAgICAgICBkdHlwZT1ucC51aW50OCwKICAgICAgICAgICAgc2hhcGU9KG5faWZnLCksCiAgICAgICAgKQogICAgICAgIGdyaWRfcGhhc2VbOl0gPSAwCiAgICAgICAgZ3JpZF9sb3dwYXNzWzpdID0gMAogICAgICAgIGRvbmVbOl0gPSAwCiAgICAgICAgZ3JpZF9waGFzZS5mbHVzaCgpCiAgICAgICAgZ3JpZF9sb3dwYXNzLmZsdXNoKCkKICAgICAgICBkb25lLmZsdXNoKCkKICAgIGVsc2U6CiAgICAgICAgZ3JpZF9waGFzZSA9IG5wLmxpYi5mb3JtYXQub3Blbl9tZW1tYXAocGhhc2VfcGF0aCwgbW9kZT0icisiKQogICAgICAgIGdyaWRfbG93cGFzcyA9IG5wLmxpYi5mb3JtYXQub3Blbl9tZW1tYXAobG93X3BhdGgsIG1vZGU9InIrIikKICAgICAgICBkb25lID0gbnAubGliLmZvcm1hdC5vcGVuX21lbW1hcChkb25lX3BhdGgsIG1vZGU9InIrIikKCiAgICBjb21wbGV0ZWRfaW5pdGlhbCA9IGludChucC5jb3VudF9ub256ZXJvKGRvbmUpKQogICAgZ3JpZF9zdGFydGVkID0gdGltZS5wZXJmX2NvdW50ZXIoKQoKICAgIG1ldGFkYXRhID0gewogICAgICAgICJzaWduYXR1cmUiOiBzaWduYXR1cmUsCiAgICAgICAgInN0YXR1cyI6ICJydW5uaW5nIiwKICAgICAgICAibl9wcyI6IGludChuX3BzKSwKICAgICAgICAibl9pZmciOiBpbnQobl9pZmcpLAogICAgICAgICJuX2kiOiBpbnQobl9pKSwKICAgICAgICAibl9qIjogaW50KG5faiksCiAgICAgICAgIm5fZ3JpZF9wcyI6IGludChuX2dyaWRfcHMpLAogICAgICAgICJ0b3RhbF93aW5kb3dzIjogaW50KHdpbmRvd3Muc2hhcGVbMF0pLAogICAgICAgICJhY3RpdmVfd2luZG93cyI6IGludChhY3RpdmVfaW5kaWNlcy5zaXplKSwKICAgICAgICAiaWZnX2JhdGNoIjogaW50KGlmZ19iYXRjaCksCiAgICAgICAgIndpbmRvd19iYXRjaCI6IGludCh3aW5kb3dfYmF0Y2gpLAogICAgICAgICJmZnRfd29ya2VycyI6IGludChmZnRfd29ya2VycyksCiAgICAgICAgImNvbXBsZXRlZCI6IGludChjb21wbGV0ZWRfaW5pdGlhbCksCiAgICAgICAgInJlc3VtZSI6IGJvb2woY2FuX3Jlc3VtZSksCiAgICAgICAgInVwZGF0ZWRfZXBvY2hfc2VjIjogdGltZS50aW1lKCksCiAgICB9CiAgICBfc3RhZ2U2X2F0b21pY19qc29uKG1ldGFfcGF0aCwgbWV0YWRhdGEpCgogICAgcHJpbnQoCiAgICAgICAgIltTVEFHRTZfU0JBU11bR1JJRF9WMl0gIgogICAgICAgIGYid2luZG93cz17d2luZG93cy5zaGFwZVswXX0sIGFjdGl2ZT17YWN0aXZlX2luZGljZXMuc2l6ZX0sICIKICAgICAgICBmImlmZ19iYXRjaD17aWZnX2JhdGNofSwgd2luZG93X2JhdGNoPXt3aW5kb3dfYmF0Y2h9LCAiCiAgICAgICAgZiJmZnRfd29ya2Vycz17ZmZ0X3dvcmtlcnN9LCByZXN1bWVkPXtjb21wbGV0ZWRfaW5pdGlhbH0ve25faWZnfSIsCiAgICAgICAgZmx1c2g9VHJ1ZSwKICAgICkKCiAgICBmb3IgaWZnX3N0YXJ0IGluIHJhbmdlKDAsIG5faWZnLCBpZmdfYmF0Y2gpOgogICAgICAgIGlmZ19zdG9wID0gbWluKGlmZ19zdGFydCArIGlmZ19iYXRjaCwgbl9pZmcpCiAgICAgICAgaWYgbnAuYWxsKGRvbmVbaWZnX3N0YXJ0OmlmZ19zdG9wXSAhPSAwKToKICAgICAgICAgICAgY29udGludWUKCiAgICAgICAgdmFsdWVzID0gbnAuYXNhcnJheSgKICAgICAgICAgICAgc291cmNlW29yZGVyLCBpZmdfc3RhcnQ6aWZnX3N0b3BdLAogICAgICAgICAgICBkdHlwZT1ucC5jb21wbGV4NjQsCiAgICAgICAgKQogICAgICAgIGdyb3VwZWQgPSBucC5hZGQucmVkdWNlYXQodmFsdWVzLCBzdGFydHMsIGF4aXM9MCkKCiAgICAgICAgY3VycmVudF9iYXRjaCA9IGlmZ19zdG9wIC0gaWZnX3N0YXJ0CiAgICAgICAgZ3JpZF9zdGFjayA9IG5wLnplcm9zKAogICAgICAgICAgICAoaW50KG5faSksIGludChuX2opLCBjdXJyZW50X2JhdGNoKSwKICAgICAgICAgICAgZHR5cGU9bnAuY29tcGxleDY0LAogICAgICAgICAgICBvcmRlcj0iRiIsCiAgICAgICAgKQogICAgICAgIGZsYXRfc3RhY2sgPSBncmlkX3N0YWNrLnJlc2hhcGUoCiAgICAgICAgICAgIChpbnQobl9pKSAqIGludChuX2opLCBjdXJyZW50X2JhdGNoKSwKICAgICAgICAgICAgb3JkZXI9IkYiLAogICAgICAgICkKICAgICAgICBmbGF0X3N0YWNrW2dyb3VwX2xpbiwgOl0gPSBncm91cGVkCgogICAgICAgIGZpbHRlcmVkLCBsb3dwYXNzID0gX3N0YWdlNl9nb2xkc3RlaW5fZmlsdGVyX2RlbnNlX2JhdGNoKAogICAgICAgICAgICBncmlkX3N0YWNrLAogICAgICAgICAgICBuX3dpbj1pbnQobl93aW4pLAogICAgICAgICAgICBhbHBoYT1mbG9hdChhbHBoYSksCiAgICAgICAgICAgIGdvbGRfZmxhZz1ib29sKGdvbGRfZmxhZyksCiAgICAgICAgICAgIGZmdF93b3JrZXJzPWludChmZnRfd29ya2VycyksCiAgICAgICAgICAgIHdpbmRvd19iYXRjaD1pbnQod2luZG93X2JhdGNoKSwKICAgICAgICAgICAgd2luZG93cz13aW5kb3dzLAogICAgICAgICAgICBiYXNlX3dlaWdodD1iYXNlX3dlaWdodCwKICAgICAgICAgICAgYWN0aXZlX2luZGljZXM9YWN0aXZlX2luZGljZXMsCiAgICAgICAgKQoKICAgICAgICBzZWxlY3RlZF9maWx0ZXJlZCA9IGZpbHRlcmVkLnJlc2hhcGUoCiAgICAgICAgICAgIChpbnQobl9pKSAqIGludChuX2opLCBjdXJyZW50X2JhdGNoKSwKICAgICAgICAgICAgb3JkZXI9IkYiLAogICAgICAgIClbbnpfZmxhdCwgOl0KICAgICAgICBzZWxlY3RlZF9sb3dwYXNzID0gbG93cGFzcy5yZXNoYXBlKAogICAgICAgICAgICAoaW50KG5faSkgKiBpbnQobl9qKSwgY3VycmVudF9iYXRjaCksCiAgICAgICAgICAgIG9yZGVyPSJGIiwKICAgICAgICApW256X2ZsYXQsIDpdCgogICAgICAgIGdyaWRfcGhhc2VbOiwgaWZnX3N0YXJ0OmlmZ19zdG9wXSA9IHNlbGVjdGVkX2ZpbHRlcmVkCiAgICAgICAgZ3JpZF9sb3dwYXNzWzosIGlmZ19zdGFydDppZmdfc3RvcF0gPSBzZWxlY3RlZF9sb3dwYXNzCiAgICAgICAgZ3JpZF9waGFzZS5mbHVzaCgpCiAgICAgICAgZ3JpZF9sb3dwYXNzLmZsdXNoKCkKCiAgICAgICAgZG9uZVtpZmdfc3RhcnQ6aWZnX3N0b3BdID0gMQogICAgICAgIGRvbmUuZmx1c2goKQoKICAgICAgICBjb21wbGV0ZWQgPSBpbnQobnAuY291bnRfbm9uemVybyhkb25lKSkKICAgICAgICBlbGFwc2VkID0gdGltZS5wZXJmX2NvdW50ZXIoKSAtIGdyaWRfc3RhcnRlZAogICAgICAgIG5ld2x5X2NvbXBsZXRlZCA9IG1heCgxLCBjb21wbGV0ZWQgLSBjb21wbGV0ZWRfaW5pdGlhbCkKICAgICAgICByYXRlID0gbmV3bHlfY29tcGxldGVkIC8gZWxhcHNlZCBpZiBlbGFwc2VkID4gMCBlbHNlIDAuMAogICAgICAgIGV0YSA9IChuX2lmZyAtIGNvbXBsZXRlZCkgLyByYXRlIGlmIHJhdGUgPiAwIGVsc2UgZmxvYXQoIm5hbiIpCgogICAgICAgIG1ldGFkYXRhLnVwZGF0ZSgKICAgICAgICAgICAgewogICAgICAgICAgICAgICAgImNvbXBsZXRlZCI6IGNvbXBsZXRlZCwKICAgICAgICAgICAgICAgICJlbGFwc2VkX3NlYyI6IGVsYXBzZWQsCiAgICAgICAgICAgICAgICAiZXRhX3NlYyI6IGV0YSwKICAgICAgICAgICAgICAgICJ1cGRhdGVkX2Vwb2NoX3NlYyI6IHRpbWUudGltZSgpLAogICAgICAgICAgICB9CiAgICAgICAgKQogICAgICAgIF9zdGFnZTZfYXRvbWljX2pzb24obWV0YV9wYXRoLCBtZXRhZGF0YSkKCiAgICAgICAgaWYgcHJvZ3Jlc3M6CiAgICAgICAgICAgIHByaW50KAogICAgICAgICAgICAgICAgIltTVEFHRTZfU0JBU11bR1JJRF9WMl0gIgogICAgICAgICAgICAgICAgZiJ7Y29tcGxldGVkfS97bl9pZmd9ICIKICAgICAgICAgICAgICAgIGYiKHsxMDAuMCAqIGNvbXBsZXRlZCAvIG5faWZnOi4xZn0lKSwgIgogICAgICAgICAgICAgICAgZiJlbGFwc2VkPXtlbGFwc2VkOi4xZn1zLCBldGE9e2V0YTouMWZ9cyIsCiAgICAgICAgICAgICAgICBmbHVzaD1UcnVlLAogICAgICAgICAgICApCgogICAgICAgIGRlbCB2YWx1ZXMsIGdyb3VwZWQsIGdyaWRfc3RhY2ssIGZsYXRfc3RhY2sKICAgICAgICBkZWwgZmlsdGVyZWQsIGxvd3Bhc3MKICAgICAgICBkZWwgc2VsZWN0ZWRfZmlsdGVyZWQsIHNlbGVjdGVkX2xvd3Bhc3MKCiAgICBtZXRhZGF0YS51cGRhdGUoCiAgICAgICAgewogICAgICAgICAgICAic3RhdHVzIjogImNvbXBsZXRlZCIsCiAgICAgICAgICAgICJjb21wbGV0ZWQiOiBpbnQobl9pZmcpLAogICAgICAgICAgICAiZWxhcHNlZF9zZWMiOiB0aW1lLnBlcmZfY291bnRlcigpIC0gZ3JpZF9zdGFydGVkLAogICAgICAgICAgICAiZXRhX3NlYyI6IDAuMCwKICAgICAgICAgICAgInVwZGF0ZWRfZXBvY2hfc2VjIjogdGltZS50aW1lKCksCiAgICAgICAgfQogICAgKQogICAgX3N0YWdlNl9hdG9taWNfanNvbihtZXRhX3BhdGgsIG1ldGFkYXRhKQoKICAgIHJldHVybiAoCiAgICAgICAgZ3JpZF9waGFzZSwKICAgICAgICBncmlkX2xvd3Bhc3MsCiAgICAgICAgbnpfZmxhdCwKICAgICAgICBuel9pLAogICAgICAgIG56X2osCiAgICAgICAgbl9ncmlkX3BzLAogICAgICAgIG1ldGFkYXRhLAogICAgKQo=').decode("utf-8")
TEST_CONTENT = base64.b64decode('ZnJvbSBfX2Z1dHVyZV9fIGltcG9ydCBhbm5vdGF0aW9ucwoKaW1wb3J0IG51bXB5IGFzIG5wCgppbXBvcnQgcHlzdGFtcHMucGlwZWxpbmUucG9ydGVkIGFzIHBvcnRlZApmcm9tIHB5c3RhbXBzLnBpcGVsaW5lLnN0YWdlNl9zYmFzIGltcG9ydCAoCiAgICBfc3RhZ2U2X2dvbGRzdGVpbl9maWx0ZXJfZGVuc2VfYmF0Y2gsCikKCgpkZWYgdGVzdF9zdGFnZTZfZ3JpZF9iYXRjaF9tYXRjaGVzX2xlZ2FjeV93cmFwX2ZpbHRfZ2xvYmFsKCkgLT4gTm9uZToKICAgIHJuZyA9IG5wLnJhbmRvbS5kZWZhdWx0X3JuZygyMDI2MDcyNykKICAgIG5faSwgbl9qLCBuX2lmZyA9IDcyLCA4OCwgMwoKICAgIHN0YWNrID0gbnAuemVyb3MoKG5faSwgbl9qLCBuX2lmZyksIGR0eXBlPW5wLmNvbXBsZXg2NCkKICAgIG9jY3VwaWVkID0gcm5nLnJhbmRvbSgobl9pLCBuX2opKSA8IDAuMDgKICAgIHBoYXNlcyA9IHJuZy51bmlmb3JtKC1ucC5waSwgbnAucGksIHNpemU9KG5faSwgbl9qLCBuX2lmZykpCiAgICBzdGFja1tvY2N1cGllZCwgOl0gPSBucC5leHAoMWogKiBwaGFzZXNbb2NjdXBpZWQsIDpdKS5hc3R5cGUobnAuY29tcGxleDY0KQoKICAgIGFjdHVhbCwgYWN0dWFsX2xvdyA9IF9zdGFnZTZfZ29sZHN0ZWluX2ZpbHRlcl9kZW5zZV9iYXRjaCgKICAgICAgICBzdGFjaywKICAgICAgICBuX3dpbj0zMiwKICAgICAgICBhbHBoYT0wLjgsCiAgICAgICAgZ29sZF9mbGFnPVRydWUsCiAgICAgICAgZmZ0X3dvcmtlcnM9MSwKICAgICAgICB3aW5kb3dfYmF0Y2g9NSwKICAgICkKCiAgICBmb3IgaW5kZXggaW4gcmFuZ2Uobl9pZmcpOgogICAgICAgIGV4cGVjdGVkLCBleHBlY3RlZF9sb3cgPSBwb3J0ZWQuX3dyYXBfZmlsdF9nbG9iYWwoCiAgICAgICAgICAgIHN0YWNrWzosIDosIGluZGV4XSwKICAgICAgICAgICAgbl93aW49MzIsCiAgICAgICAgICAgIGFscGhhPTAuOCwKICAgICAgICAgICAgbG93X2ZsYWc9InkiLAogICAgICAgICkKICAgICAgICBucC50ZXN0aW5nLmFzc2VydF9hbGxjbG9zZSgKICAgICAgICAgICAgYWN0dWFsWzosIDosIGluZGV4XSwKICAgICAgICAgICAgZXhwZWN0ZWQsCiAgICAgICAgICAgIHJ0b2w9MmUtNiwKICAgICAgICAgICAgYXRvbD0yZS02LAogICAgICAgICkKICAgICAgICBucC50ZXN0aW5nLmFzc2VydF9hbGxjbG9zZSgKICAgICAgICAgICAgYWN0dWFsX2xvd1s6LCA6LCBpbmRleF0sCiAgICAgICAgICAgIGV4cGVjdGVkX2xvdywKICAgICAgICAgICAgcnRvbD0yZS02LAogICAgICAgICAgICBhdG9sPTJlLTYsCiAgICAgICAgKQo=').decode("utf-8")
RUNNER_CONTENT = base64.b64decode('IyEvdXNyL2Jpbi9lbnYgYmFzaApzZXQgLWV1byBwaXBlZmFpbAoKUk9PVD0iJHtQWVNUQU1QU19ST09UOi0vaG9tZS91YnVudHUvc29mdHdhcmUvcHlzdGFtcHMtbWFpbn0iCkRBVEFTRVQ9IiR7UkVBTF9EQVRBU0VUOi0vbW50L3ZvbC1nZGMyOG4xci9pbnNhci9jYW5nemhvdV9QNjkvcHlzdGFtcHNfc2Jhc19wc19vcHRpbWl6ZWR9IgoKZXhwb3J0IFBZVEhPTlVOQlVGRkVSRUQ9MQpleHBvcnQgT01QX05VTV9USFJFQURTPSIke09NUF9OVU1fVEhSRUFEUzotMX0iCmV4cG9ydCBPUEVOQkxBU19OVU1fVEhSRUFEUz0iJHtPUEVOQkxBU19OVU1fVEhSRUFEUzotMX0iCmV4cG9ydCBNS0xfTlVNX1RIUkVBRFM9IiR7TUtMX05VTV9USFJFQURTOi0xfSIKZXhwb3J0IE5VTUVYUFJfTlVNX1RIUkVBRFM9IiR7TlVNRVhQUl9OVU1fVEhSRUFEUzotMX0iCmV4cG9ydCBCTElTX05VTV9USFJFQURTPSIke0JMSVNfTlVNX1RIUkVBRFM6LTF9IgpleHBvcnQgTUFMTE9DX0FSRU5BX01BWD0iJHtNQUxMT0NfQVJFTkFfTUFYOi0yfSIKCmV4cG9ydCBQWVNUQU1QU19TQkFTX1BST0dSRVNTPSIke1BZU1RBTVBTX1NCQVNfUFJPR1JFU1M6LTF9IgpleHBvcnQgUFlTVEFNUFNfU0JBU19FREdFX0NIVU5LPSIke1BZU1RBTVBTX1NCQVNfRURHRV9DSFVOSzotMjA0OH0iCgpleHBvcnQgUFlTVEFNUFNfU1RBR0U2X0dSSURfSUZHX0JBVENIPSIke1BZU1RBTVBTX1NUQUdFNl9HUklEX0lGR19CQVRDSDotNH0iCmV4cG9ydCBQWVNUQU1QU19TVEFHRTZfR1JJRF9XSU5ET1dfQkFUQ0g9IiR7UFlTVEFNUFNfU1RBR0U2X0dSSURfV0lORE9XX0JBVENIOi0zMn0iCmV4cG9ydCBQWVNUQU1QU19TVEFHRTZfR1JJRF9GRlRfV09SS0VSUz0iJHtQWVNUQU1QU19TVEFHRTZfR1JJRF9GRlRfV09SS0VSUzotMTZ9IgpleHBvcnQgUFlTVEFNUFNfU1RBR0U2X0dSSURfUkVTVU1FPSIke1BZU1RBTVBTX1NUQUdFNl9HUklEX1JFU1VNRTotMX0iCgpleHBvcnQgUFlTVEFNUFNfU1RBR0U2X1NOQVBIVV9XT1JLRVJTPSIke1BZU1RBTVBTX1NUQUdFNl9TTkFQSFVfV09SS0VSUzotNH0iCmV4cG9ydCBQWVNUQU1QU19TQkFTX0FOTkVBTF9XT1JLRVJTPSIke1BZU1RBTVBTX1NCQVNfQU5ORUFMX1dPUktFUlM6LTh9IgpleHBvcnQgUFlTVEFNUFNfU0JBU19BTk5FQUxfUlVOUz0iJHtQWVNUQU1QU19TQkFTX0FOTkVBTF9SVU5TOi0xNX0iCmV4cG9ydCBQWVNUQU1QU19TQkFTX1NUUklDVF9BTk5FQUw9IiR7UFlTVEFNUFNfU0JBU19TVFJJQ1RfQU5ORUFMOi0xfSIKZXhwb3J0IFBZU1RBTVBTX1NCQVNfS0VFUF9XT1JLPSIke1BZU1RBTVBTX1NCQVNfS0VFUF9XT1JLOi0wfSIKCmNkICIkUk9PVCIKCmVjaG8gIj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSIKZWNobyAiU3RhTVBTLWNvbXBhdGlibGUgU0JBUyBTdGFnZSA2IOKAlCBHUklEIGJhdGNoIHYyIgplY2hvICJEYXRhc2V0ICAgICAgICAgIDogJERBVEFTRVQiCmVjaG8gIkdSSUQgSUZHIGJhdGNoICAgOiAkUFlTVEFNUFNfU1RBR0U2X0dSSURfSUZHX0JBVENIIgplY2hvICJHUklEIHdpbmRvdyBiYXRjaDogJFBZU1RBTVBTX1NUQUdFNl9HUklEX1dJTkRPV19CQVRDSCIKZWNobyAiR1JJRCBGRlQgd29ya2VycyA6ICRQWVNUQU1QU19TVEFHRTZfR1JJRF9GRlRfV09SS0VSUyIKZWNobyAiR1JJRCByZXN1bWUgICAgICA6ICRQWVNUQU1QU19TVEFHRTZfR1JJRF9SRVNVTUUiCmVjaG8gIkVkZ2UgY2h1bmsgICAgICAgOiAkUFlTVEFNUFNfU0JBU19FREdFX0NIVU5LIgplY2hvICJTTkFQSFUgd29ya2VycyAgIDogJFBZU1RBTVBTX1NUQUdFNl9TTkFQSFVfV09SS0VSUyIKZWNobyAiQW5uZWFsIHdvcmtlcnMgICA6ICRQWVNUQU1QU19TQkFTX0FOTkVBTF9XT1JLRVJTIgplY2hvICJBbm5lYWwgcnVucyAgICAgIDogJFBZU1RBTVBTX1NCQVNfQU5ORUFMX1JVTlMiCmVjaG8gIlN0cmljdCBhbm5lYWwgICAgOiAkUFlTVEFNUFNfU0JBU19TVFJJQ1RfQU5ORUFMIgplY2hvICI9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0iCgpleGVjIHB5dGhvbiAtbSBweXN0YW1wcy5waXBlbGluZS5zdGFnZTZfc2JhcyBcCiAgICAtLWRhdGFzZXQgIiREQVRBU0VUIiBcCiAgICAtLWlvLXdvcmtlcnMgMSBcCiAgICAiJEAiCg==').decode("utf-8")
ROLLBACK_CONTENT = base64.b64decode('IyEvdXNyL2Jpbi9lbnYgYmFzaApzZXQgLWV1byBwaXBlZmFpbApST09UPSIke1BZU1RBTVBTX1JPT1Q6LS9ob21lL3VidW50dS9zb2Z0d2FyZS9weXN0YW1wcy1tYWlufSIKUE9JTlRFUj0iJFJPT1QvLnN0YWdlNl9ncmlkX3JlZmFjdG9yX2xhc3RfYmFja3VwIgpbWyAtcyAiJFBPSU5URVIiIF1dIHx8IHsgZWNobyAi5rKh5pyJ5om+5Yiw5aSH5Lu95oyH6ZKI77yaJFBPSU5URVIiID4mMjsgZXhpdCAyOyB9CkJBQ0tVUD0iJChjYXQgIiRQT0lOVEVSIikiCk1PRFVMRT0iJFJPT1QvcHlzdGFtcHMvcGlwZWxpbmUvc3RhZ2U2X3NiYXMucHkiCltbIC1mICIkQkFDS1VQL3B5c3RhbXBzL3BpcGVsaW5lL3N0YWdlNl9zYmFzLnB5LmN1cnJlbnQiIF1dIHx8IHsKICAgIGVjaG8gIuWkh+S7veS4reayoeacieWOn3N0YWdlNl9zYmFzLnB577yaJEJBQ0tVUCIgPiYyCiAgICBleGl0IDMKfQpjcCAtYSAiJEJBQ0tVUC9weXN0YW1wcy9waXBlbGluZS9zdGFnZTZfc2Jhcy5weS5jdXJyZW50IiAiJE1PRFVMRSIKW1sgLWYgIiRCQUNLVVAvcnVuX3N0YWdlNl9mYXN0LnNoIiBdXSAmJiBjcCAtYSAiJEJBQ0tVUC9ydW5fc3RhZ2U2X2Zhc3Quc2giICIkUk9PVC9ydW5fc3RhZ2U2X2Zhc3Quc2giCnB5dGhvbiAtbSBweV9jb21waWxlICIkTU9EVUxFIgplY2hvICLlt7Llm57mu5rliLDvvJokQkFDS1VQIgo=').decode("utf-8")

ROOT = Path(os.environ.get("PYSTAMPS_ROOT", "/home/ubuntu/software/pystamps-main")).expanduser().resolve()
DATASET = Path(os.environ.get("REAL_DATASET", "/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized")).expanduser().resolve()
MODULE = ROOT / "pystamps/pipeline/stage6_sbas.py"
RUNNER = ROOT / "run_stage6_fast.sh"
TEST_FILE = ROOT / "tests/test_stage6_sbas_grid_batch.py"

def stage6_pids() -> list[int]:
    query = r"[p]ython.*pystamps\.pipeline\.stage6_sbas|[r]un_stage6_sbas\.sh|[r]un_stage6_fast\.sh"
    result = subprocess.run(["pgrep", "-f", query], text=True, capture_output=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.strip() or "pgrep failed")
    return [int(line) for line in result.stdout.splitlines() if line.strip()]

def stop_stage6() -> None:
    pids = stage6_pids()
    if not pids:
        return
    if os.environ.get("FORCE_STOP", "0") != "1":
        subprocess.run(["ps", "-fp", ",".join(str(pid) for pid in pids)], check=False)
        raise SystemExit(
            "检测到Stage 6仍在运行。确认停止后执行：\n"
            f"  FORCE_STOP=1 python {Path(__file__).name}"
        )
    print("停止Stage 6进程：", pids)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(8)
    for pid in stage6_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

def valid_source(path: Path) -> bool:
    anchors = (
        "def stage6_sbas_unwrap(",
        "ph_in = ph_w[:, unwrap_ix].astype(np.complex64)",
        "with ThreadPoolExecutor(max_workers=grid_workers) as pool:",
        "grid_phase = np.zeros((n_grid_ps, unwrap_ix.size), dtype=np.complex64)",
    )
    if not path.is_file() or path.stat().st_size < 10000:
        return False
    try:
        text = path.read_text(encoding="utf-8")
        compile(text, str(path), "exec")
    except Exception:
        return False
    return all(anchor in text for anchor in anchors)

def restore_source() -> Path:
    candidates = [MODULE, Path(str(MODULE) + ".bak_stage6_opt")]
    candidates.extend(sorted(
        ROOT.glob(".stage6_sbas_backup/*/pystamps/pipeline/stage6_sbas.py"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    ))
    candidates.extend(sorted(
        ROOT.glob("pystamps/pipeline/stage6_sbas.py.*"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    ))
    source = next((path for path in candidates if valid_source(path)), None)
    if source is None:
        raise SystemExit(
            "没有找到语法正确且包含原始GRID代码的stage6_sbas.py。"
            "请先重新执行原始apply_stage6_sbas_patch.sh。"
        )
    if source.resolve() != MODULE.resolve():
        shutil.copy2(source, MODULE)
        print("已从有效备份恢复：", source)
    else:
        print("当前源码有效：", MODULE)
    return source

def apply_refactor() -> None:
    text = MODULE.read_text(encoding="utf-8")
    marker = "# === STAGE6_SBAS_GRID_BATCH_V2 ==="
    if marker in text:
        print("GRID v2代码已经存在，跳过重复插入。")
        return

    if "import hashlib\n" not in text:
        text = text.replace("import argparse\n", "import argparse\nimport hashlib\n", 1)
    if "from scipy import fft as scipy_fft" not in text:
        text = text.replace(
            "import numpy as np\n",
            "import numpy as np\n\nfrom scipy import fft as scipy_fft\nfrom scipy import ndimage\n",
            1,
        )

    anchor = "\ndef stage6_sbas_unwrap(\n"
    if anchor not in text:
        raise SystemExit("未找到stage6_sbas_unwrap插入位置。")
    text = text.replace(anchor, "\n" + HELPER.rstrip() + "\n\n" + anchor.lstrip(), 1)

    old_work = "    if work_dir.exists():\n        shutil.rmtree(work_dir)\n    work_dir.mkdir(parents=True)\n"
    new_work = "    grid_resume = _env_bool(\"PYSTAMPS_STAGE6_GRID_RESUME\", True)\n    if work_dir.exists() and not grid_resume:\n        shutil.rmtree(work_dir)\n    work_dir.mkdir(parents=True, exist_ok=True)\n"
    if old_work not in text:
        raise SystemExit("未找到work_dir初始化块。")
    text = text.replace(old_work, new_work, 1)

    old_pm = "        pm2 = read_mat(root / \"pm2.mat\")\n        bp2 = read_mat_variables(root / \"bp2.mat\", (\"bperp_mat\",))\n"
    new_pm = (
        "        unwrap_patch_phase = _mat_text(\n"
        "            parms.get(\"unwrap_patch_phase\"),\n"
        "            \"n\",\n"
        "        ).lower() == \"y\"\n"
        "        pm2_variables = (\n"
        "            (\"K_ps\", \"ph_patch\")\n"
        "            if unwrap_patch_phase\n"
        "            else (\"K_ps\",)\n"
        "        )\n"
        "        pm2 = read_mat_variables(root / \"pm2.mat\", pm2_variables)\n"
        "        bp2 = read_mat_variables(root / \"bp2.mat\", (\"bperp_mat\",))\n"
    )
    if old_pm not in text:
        raise SystemExit("未找到pm2读取块。")
    text = text.replace(old_pm, new_pm, 1)

    duplicate_flag = (
        "        unwrap_patch_phase = _mat_text(\n"
        "            parms.get(\"unwrap_patch_phase\"),\n"
        "            \"n\",\n"
        "        ).lower() == \"y\"\n"
        "        phase_restore = np.zeros((n_ps, n_ifg), dtype=np.float32)\n"
    )
    if duplicate_flag not in text:
        raise SystemExit("未找到unwrap_patch_phase/phase_restore块。")
    text = text.replace(duplicate_flag, "        phase_restore: np.ndarray | None = None\n", 1)

    old = "                phase_restore += correction\n"
    new = (
        "                if phase_restore is None:\n"
        "                    phase_restore = np.zeros((n_ps, n_ifg), dtype=np.float32)\n"
        "                phase_restore += correction\n"
    )
    if old not in text:
        raise SystemExit("未找到SCLA correction恢复块。")
    text = text.replace(old, new, 1)

    old = "                    phase_restore += ramp_arr\n"
    new = (
        "                    if phase_restore is None:\n"
        "                        phase_restore = np.zeros((n_ps, n_ifg), dtype=np.float32)\n"
        "                    phase_restore += ramp_arr\n"
    )
    if old not in text:
        raise SystemExit("未找到ramp恢复块。")
    text = text.replace(old, new, 1)

    old_normalize = "        ph_w = _normalize_complex(ph_w)\n"
    new_normalize = (
        "        ph_w = np.asarray(ph_w, dtype=np.complex64)\n"
        "        ph_w_magnitude = np.abs(ph_w)\n"
        "        np.divide(\n"
        "            ph_w,\n"
        "            ph_w_magnitude,\n"
        "            out=ph_w,\n"
        "            where=ph_w_magnitude != 0,\n"
        "        )\n"
        "        del ph_w_magnitude\n"
    )
    if old_normalize not in text:
        raise SystemExit("未找到ph_w归一化语句。")
    text = text.replace(old_normalize, new_normalize, 1)

    grid_start_token = "        ph_in = ph_w[:, unwrap_ix].astype(np.complex64)\n"
    grid_end_token = "        nzix = nz_flat.reshape((n_i, n_j), order=\"F\")\n"
    grid_start = text.index(grid_start_token)
    grid_end = text.index(grid_end_token, grid_start)

    grid_replacement = (
        "        if (\n"
        "            unwrap_ix.size == n_ifg\n"
        "            and np.array_equal(\n"
        "                unwrap_ix,\n"
        "                np.arange(n_ifg, dtype=np.int64),\n"
        "            )\n"
        "        ):\n"
        "            ph_in = ph_w\n"
        "        else:\n"
        "            ph_in = np.ascontiguousarray(\n"
        "                ph_w[:, unwrap_ix],\n"
        "                dtype=np.complex64,\n"
        "            )\n\n"
        "        lin0 = (\n"
        "            (grid_j - 1) * n_i\n"
        "            + (grid_i - 1)\n"
        "        ).astype(np.int64)\n\n"
        "        (\n"
        "            grid_phase,\n"
        "            grid_lowpass,\n"
        "            nz_flat,\n"
        "            nz_i,\n"
        "            nz_j,\n"
        "            n_grid_ps,\n"
        "            grid_meta,\n"
        "        ) = _stage6_build_grid_batched(\n"
        "            ph_in=ph_in,\n"
        "            lin0=lin0,\n"
        "            n_i=n_i,\n"
        "            n_j=n_j,\n"
        "            n_win=prefilt_win,\n"
        "            alpha=gold_alpha,\n"
        "            gold_flag=gold_flag,\n"
        "            work_dir=work_dir,\n"
        "            progress=progress,\n"
        "        )\n\n"
        "        print(\n"
        "            \"[STAGE6_SBAS] \"\n"
        "            f\"n_ps={n_ps}, n_ifg={n_ifg}, selected_ifg={unwrap_ix.size}, \"\n"
        "            f\"n_image={day.size}, grid={n_i}x{n_j}, grid_ps={n_grid_ps}, \"\n"
        "            f\"network_source={network_source}, \"\n"
        "            f\"grid_active_windows={grid_meta['active_windows']}\",\n"
        "            flush=True,\n"
        "        )\n\n"
    )
    text = text[:grid_start] + grid_replacement + text[grid_end:]

    old_restore = "            ph_uw_selected[valid, :] += phase_restore[valid, :][:, unwrap_ix]\n"
    new_restore = (
        "            if phase_restore is not None:\n"
        "                ph_uw_selected[valid, :] += (\n"
        "                    phase_restore[valid, :][:, unwrap_ix]\n"
        "                )\n"
    )
    if old_restore not in text:
        raise SystemExit("未找到phase_restore最终恢复语句。")
    text = text.replace(old_restore, new_restore, 1)

    MODULE.write_text(text, encoding="utf-8")
    py_compile.compile(str(MODULE), doraise=True)
    print("Stage 6 GRID v2源码重构和语法检查完成。")

def main() -> int:
    print("=" * 88)
    print("Stage 6 SBAS GRID code-level refactor v2")
    print("ROOT   :", ROOT)
    print("DATASET:", DATASET)
    print("=" * 88)

    stop_stage6()
    if not (ROOT / "pystamps/pipeline").is_dir():
        raise SystemExit(f"不是有效pySTAMPS仓库：{ROOT}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = ROOT / ".stage6_grid_refactor_backup" / stamp
    (backup / "pystamps/pipeline").mkdir(parents=True, exist_ok=True)
    (backup / "tests").mkdir(parents=True, exist_ok=True)
    if MODULE.exists():
        shutil.copy2(MODULE, backup / "pystamps/pipeline/stage6_sbas.py.current")
    if RUNNER.exists():
        shutil.copy2(RUNNER, backup / "run_stage6_fast.sh")
    if TEST_FILE.exists():
        shutil.copy2(TEST_FILE, backup / "tests/test_stage6_sbas_grid_batch.py")
    (ROOT / ".stage6_grid_refactor_last_backup").write_text(str(backup), encoding="utf-8")

    restore_source()
    apply_refactor()

    TEST_FILE.write_text(TEST_CONTENT, encoding="utf-8")
    RUNNER.write_text(RUNNER_CONTENT, encoding="utf-8")
    RUNNER.chmod(0o755)
    rollback = ROOT / "rollback_stage6_grid_refactor.sh"
    rollback.write_text(ROLLBACK_CONTENT, encoding="utf-8")
    rollback.chmod(0o755)

    if os.environ.get("SKIP_TESTS", "0") != "1":
        subprocess.run([sys.executable, "-m", "pytest", "-q", str(TEST_FILE)], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "-m", "pytest", "-q", str(ROOT / "tests/test_stage6_ported.py")], cwd=ROOT, check=True)

    old_work = DATASET / "_stage6_sbas_work"
    if old_work.is_dir():
        partial = DATASET / "_stage6_partial_backup" / f"grid_v1_{stamp}"
        partial.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_work), str(partial))
        print("旧GRID临时目录已移动到：", partial)

    print("=" * 88)
    print("重构完成")
    print("备份：", backup)
    print("启动：", RUNNER)
    print("回滚：", rollback)
    print("=" * 88)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
