#!/usr/bin/env python3
#
#generate imports
import mingus.core.notes as notes
from mingus.containers import Note, NoteContainer, Bar, Track, Instrument
import random
import mingus.extra.lilypond as LilyPond

#Transform imports
import subprocess
import os
from PIL import Image

from functools import reduce
import numpy as np
import cv2

# script handling
import getopt
import sys
#stałe globalne

allNotesM = ["A-3", "B-3", "C-4", "D-4","E-4", "F-4", "G-4", "A-4", "B-4", "C-5", "D-5", "E-5", "F-5", "G-5", "A-5", "B-5", "C-6" ]
lenAllNotesM = len(allNotesM)
largestInterval = 4
pOfChromatics=.05

quarterGroupOptions16 = [[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[.5,.5],[.5,.5],[.5,.5],[.5,.5],[.5,.5],[.5,.5], [.5, .5], [.25, .25, .5], [.5, .25, .25], [.25, .25, .25, .25]]
quarterGroupOptions8 = [[1],[1],[1],[0.5,0.5]]
bar4GroupOptions = [[4], [2,2], [2,1,1],[2,1,1],[1,1,2],[1,1,2],[1,1,1,1],[1,1,1,1]]
bar3GroupOptions = [[2,1],[1,2],[1,1,1]]

pOfRests=.1


# jeśli before=-1 -> pierwsza nuta
def newNoteIndexM(before):
    if before==-1:
        return random.randint(0,lenAllNotesM-1)

    if before<largestInterval:
        return random.randint(0,2*largestInterval)

    if before>lenAllNotesM-largestInterval-1:
        return random.randint(lenAllNotesM-2*largestInterval, lenAllNotesM-1)

    return random.randint(before-largestInterval,before+largestInterval-1)

# dla length>0
def newNoteIndexListM(length):
    prev = newNoteIndexM(-1)
    melody = [prev]
    for i in range(1,length):
        prev = newNoteIndexM(prev)
        melody.append(prev)
    return melody


def newMelodyWithoutChromatics(length):
    return [Note(allNotesM[a]) for a in newNoteIndexListM(length)]

def newMelody(length):
    melody = []
    for index in newNoteIndexListM(length):
        k = random.random()
        note = Note(allNotesM[index])
        if k<pOfChromatics:
            note.augment()
        elif k>1-pOfChromatics:
            note.diminish()
        melody.append(note)
    return melody

def newQuarterGroup(with16):
    if with16:
        return random.choice(quarterGroupOptions16)
    else:
        return random.choice(quarterGroupOptions8)

def newBarRhythm(beats,with16):
    finalRhythm=[]
    if beats==4:
        rhythm = random.choice(bar4GroupOptions)
    if beats==3:
        rhythm = random.choice(bar3GroupOptions)

    for ii in range(len(rhythm)):
        if rhythm[ii]==1:
            finalRhythm.extend(newQuarterGroup(with16))
        else:
            finalRhythm.append(rhythm[ii])
    return finalRhythm

# NewTrack(liczba_uderzeń_w_takcie, liczba_taktów, czy_z_chromatyką, czy_z_16)
def NewTrack(beats,count,withChromatics,with16):
    track=Track(Instrument())
    rhythms=[]
    noOfNotes=0
    melodyCount=0

    for ii in range(count):
        rhythms.append(newBarRhythm(beats,with16))
        noOfNotes+=len(rhythms[ii])

    if withChromatics:
        melody = newMelody(noOfNotes)
    else:
        melody = newMelodyWithoutChromatics(noOfNotes)

    for rhythm in rhythms:
        b = Bar('C',(beats,4))
        for note in rhythm:
            k=random.random()
            if k>pOfRests:
                b.place_notes(melody[melodyCount], 4/note)
            else:
                b.place_notes(None, 4/note)
            melodyCount+=1
        track+b
    return track

def CleanTrack(track):
    delete_clef_string = " \n \override Staff.Clef.color = #white \n \override Staff.Clef.layer = #-1"
    delete_time_string = " \n \override Staff.TimeSignature.color = #white \n \override Staff.TimeSignature.layer = #-1"
    track = track[0] + delete_clef_string + delete_time_string+ track[1:]
    return track

def GenerateNewCleanTrack(beats=4,count=10,withChromatics=False,with16=False):
    track = NewTrack(beats,count,withChromatics,with16)
    lp = LilyPond.from_Track(track)
    return CleanTrack(lp)


#### Transform

def GenerateCropped(ly_string, filename, command='-fpng'):
    """Generates cropped PNG it is slightly changed version of minugs save_string_and_execute_LilyPond function"""
    ly_string = '\\version "2.10.33"\n' + ly_string
    if filename[-4] in ['.pdf' or '.png']:
        filename = filename[:-4]
    try:
        f = open(filename + '.ly', 'w')
        f.write(ly_string)
        f.close()
    except:
        return False
    command = 'lilypond -dresolution=300 -dpreview %s -o "%s" "%s.ly"' % (command, filename, filename)
    p = subprocess.Popen(command, shell=True).wait()
    os.remove(filename + '.ly')
    return True

def imgConvert(from_name, to_name):
    im = Image.open(from_name)
    rgb_im = im.convert('RGB')
    rgb_im.save(to_name)

def getRotationMatrixManual(rotation_angles):
    rotation_angles = [ np.deg2rad(x) for x in rotation_angles]

    phi         = rotation_angles[0] # around x
    gamma       = rotation_angles[1] # around y
    theta       = rotation_angles[2] # around z

    # X rotation
    Rphi        = np.eye(4,4)
    sp          = np.sin(phi)
    cp          = np.cos(phi)
    Rphi[1,1]   = cp
    Rphi[2,2]   = Rphi[1,1]
    Rphi[1,2]   = -sp
    Rphi[2,1]   = sp

    # Y rotation
    Rgamma        = np.eye(4,4)
    sg            = np.sin(gamma)
    cg            = np.cos(gamma)
    Rgamma[0,0]   = cg
    Rgamma[2,2]   = Rgamma[0,0]
    Rgamma[0,2]   = sg
    Rgamma[2,0]   = -sg

    # Z rotation (in-image-plane)
    Rtheta      = np.eye(4,4)
    st          = np.sin(theta)
    ct          = np.cos(theta)
    Rtheta[0,0] = ct
    Rtheta[1,1] = Rtheta[0,0]
    Rtheta[0,1] = -st
    Rtheta[1,0] = st

    R           = reduce(lambda x,y : np.matmul(x,y), [Rphi, Rgamma, Rtheta])

    return R

def pointForCV(ptsIn, ptsOut, W, H, sidelength):

    ptsIn2D      =  ptsIn[0,:]
    ptsOut2D     =  ptsOut[0,:]
    ptsOut2Dlist =  []
    ptsIn2Dlist  =  []

    for i in range(0,4):
        ptsOut2Dlist.append([ptsOut2D[i,0], ptsOut2D[i,1]])
        ptsIn2Dlist.append([ptsIn2D[i,0], ptsIn2D[i,1]])

    pin  =  np.array(ptsIn2Dlist)   +  [W/2.,H/2.]
    pout = (np.array(ptsOut2Dlist)  +  [1.,1.]) * (0.5*sidelength)
    pin  = pin.astype(np.float32)
    pout = pout.astype(np.float32)

    return pin, pout

def warpMatrix(W, H, x_angle, y_angle, z_angle,fV):

    # M is to be estimated
    M          = np.eye(4, 4)

    fVhalf     = np.deg2rad(fV/2.)
    d          = np.sqrt(W*W+H*H)
    sideLength = d/np.cos(fVhalf)
    h          = d/(2.0*np.sin(fVhalf))
    n          = h-(d/2.0);
    f          = h+(d/2.0);

    # Translation
    T       = np.eye(4,4)
    T[2,3]  = -h

    # Rotation matrices around x,y,z
    R = getRotationMatrixManual([x_angle, y_angle, z_angle])


    # Projection Matrix
    P       = np.eye(4,4)
    P[0,0]  = 1.0/np.tan(fVhalf)
    P[1,1]  = P[0,0]
    P[2,2]  = -(f+n)/(f-n)
    P[2,3]  = -(2.0*f*n)/(f-n)
    P[3,2]  = -1.0

    F       = reduce(lambda x,y : np.matmul(x,y), [P, T, R])

    ptsIn = np.array([[
                 [-W/2., -H/2., 0.],[ W/2., -H/2., 0.],[ -W/2.,H/2., 0.],[W/2.,H/2., 0.]
                 ]])
    ptsOut  = np.array(np.zeros((ptsIn.shape), dtype=ptsIn.dtype))
    ptsOut  = cv2.perspectiveTransform(ptsIn, F)

    ptsInPt2f, ptsOutPt2f = pointForCV(ptsIn, ptsOut, W, H, sideLength)
    # check float32 otherwise OpenCV throws an error
    assert(ptsInPt2f.dtype  == np.float32)
    assert(ptsOutPt2f.dtype == np.float32)
    M33 = cv2.getPerspectiveTransform(ptsInPt2f,ptsOutPt2f)
    return M33, sideLength, ptsInPt2f, ptsOutPt2f

def warpImage(src,x_angle,y_angle,z_angle,fv,corners=None):
    H,W,Nc    = src.shape
    M,sl,ptsIn, ptsOut      = warpMatrix(W,H, x_angle,y_angle,z_angle,fv);              #Compute warp matrix
    sl = int(sl)
    dst = cv2.warpPerspective(src,M, (sl,sl),borderValue=[255,255,255]); #Do actual image warp
    left_right_margin = random.uniform(2,50)
    top_bot_margin = random.uniform(2,50)
    left_upper = [min([x[0] for x in ptsOut]),min([x[1] for x in ptsOut])]
    right_lower = [max([x[0] for x in ptsOut]),max([x[1] for x in ptsOut])]
    left_upper[0] = int(max(left_upper[0]-left_right_margin,0))
    left_upper[1] = int(max(left_upper[1]-top_bot_margin,0))
    right_lower[0] = int(min(right_lower[0]+left_right_margin,sl-1))
    right_lower[1] = int(min(right_lower[1]+top_bot_margin,sl-1))
    return dst[left_upper[1]:right_lower[1],left_upper[0]:right_lower[0]]

#### Data generation

def randomWarpImage(src,x_range=15,y_range=15,z_range=15):
    x_angle = int(random.uniform(-x_range,x_range))
    y_angle = int(random.uniform(-y_range,y_range))
    z_angle = int(random.uniform(-z_range,z_range))
    fov = int(random.uniform(30,50))
    warped_image = warpImage(src,x_angle,y_angle,z_angle,fov)
    return warped_image


if __name__ == '__main__':

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "x:y:z:p:t:n:")
    settings = {"n": 20,
                "x" : 5,
                "y" : 8,
                "z" : 8,
                "p" : "pictures",
                "t" : "texts"}
    int_settings = ["n", "x", "y","z"]
    for key,val in opts:
        settings[key.strip("-")] = val if key.strip("-") not in int_settings else int(val)

    if not os.path.isdir(settings["p"]):
        os.mkdir(settings["p"])
    if not os.path.isdir(settings["t"]):
        os.mkdir(settings["t"])
    print(settings)
    for i in range(settings["n"]):
        beats = 4 #random.choices([3,4],weights=[0.25,0.75], k=1)[0]
        count = random.choices([1,2,3,4,5],weights=[1,1,1,1,1], k=1)[0]
        track = NewTrack(beats,count,withChromatics=False,with16=False)
        track_string = LilyPond.from_Track(track)
        track_string_clean = CleanTrack(track_string)
        GenerateCropped(track_string_clean,"tmp_track")
        imgConvert("tmp_track.preview.png", "tmp_track.jpg")
        src   = cv2.imread('tmp_track.jpg')
        src    = src[...,::-1] # BGR to RGB
        im = randomWarpImage(src,settings["x"],settings["y"],settings["z"])
        im = Image.fromarray(im[:,150:,:])

        im.save(f"./{settings['p']}/example_{i}.jpg")
        with open(f"./{settings['t']}/example_{i}.txt", "w") as text_file:
            text_file.write(track_string)
