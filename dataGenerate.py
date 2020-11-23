#
# generate imports
import mingus.core.notes as notes
from mingus.containers import Note, NoteContainer, Bar, Track, Instrument
import random
import mingus.extra.lilypond as LilyPond

# Transform imports
import subprocess
import os
from PIL import Image
import skimage

from functools import reduce
import numpy as np
import cv2
import re


# stałe globalne
allNotesM = [
    "A-3",
    "B-3",
    "C-4",
    "D-4",
    "E-4",
    "F-4",
    "G-4",
    "A-4",
    "B-4",
    "C-5",
    "D-5",
    "E-5",
    "F-5",
    "G-5",
    "A-5",
    "B-5",
    "C-6",
]
lenAllNotesM = len(allNotesM)
largestInterval = 4
pOfChromatics = 0.05

quarterGroupOptions16 = [
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.25, 0.25, 0.5],
    [0.5, 0.25, 0.25],
    [0.25, 0.25, 0.25, 0.25],
]
quarterGroupOptions8 = [[1], [1], [0.5, 0.5]]
bar4GroupOptions = [
    [4],
    [4],
    [2, 2],
    [2, 1, 1],
    [2, 1, 1],
    [1, 1, 2],
    [1, 1, 2],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
]
bar3GroupOptions = [[2, 1], [1, 2], [1, 1, 1]]

pOfRests = 0.15
noteSymbols = [
    "\\\\marcato",
    "\\\\stopped",
    "\\\\tenuto",
    "\\\\staccatissimo",
    "\\\\accent",
    "\\\\staccato",
    "\\\\portato",
    "^\\\\ppp",
    "^\\\\pp",
    "^\\\\p",
    "^\\\\mp",
    "^\\\\mf",
    "^\\\\f",
    "^\\\\ff",
    "^\\\\fff",
    "^\\\\mp",
    "^\\\\sf",
    "^\\\\sfz",
    "_\\\\ppp",
    "_\\\\pp",
    "_\\\\p",
    "_\\\\mp",
    "_\\\\mf",
    "_\\\\f",
    "_\\\\ff",
    "_\\\\fff",
    "_\\\\mp",
    "_\\\\sf",
    "_\\\\sfz",
]

# jeśli before=-1 -> pierwsza nuta
def newNoteIndexM(before):
    if before == -1:
        return random.randint(0, lenAllNotesM - 1)

    if before < largestInterval:
        return random.randint(0, 2 * largestInterval)

    if before > lenAllNotesM - largestInterval - 1:
        return random.randint(lenAllNotesM - 2 * largestInterval, lenAllNotesM - 1)

    return random.randint(before - largestInterval, before + largestInterval - 1)


# dla length>0
def newNoteIndexListM(length):
    prev = newNoteIndexM(-1)
    melody = [prev]
    for i in range(1, length):
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
        if k < pOfChromatics:
            note.augment()
        elif k > 1 - pOfChromatics:
            note.diminish()
        melody.append(note)
    return melody


def newQuarterGroup(with16):
    if with16:
        return random.choice(quarterGroupOptions16)
    else:
        return random.choice(quarterGroupOptions8)


def newBarRhythm(beats, with16):
    finalRhythm = []
    if beats == 4:
        rhythm = random.choice(bar4GroupOptions)
    if beats == 3:
        rhythm = random.choice(bar3GroupOptions)

    for ii in range(len(rhythm)):
        if rhythm[ii] == 1:
            finalRhythm.extend(newQuarterGroup(with16))
        else:
            finalRhythm.append(rhythm[ii])
    return finalRhythm


def NewTrack(beats, count, withChromatics, with16):
    """
    NewTrack(liczba_uderzeń_w_takcie, liczba_taktów, czy_z_chromatyką, czy_z_16)
    zwraca krotkę z trackiem i liczbą nut
    """
    track = Track(Instrument())
    rhythms = []
    noOfNotes = 0
    melodyCount = 0

    for ii in range(count):
        rhythms.append(newBarRhythm(beats, with16))
        noOfNotes += len(rhythms[ii])

    if withChromatics:
        melody = newMelody(noOfNotes)
    else:
        melody = newMelodyWithoutChromatics(noOfNotes)

    for rhythm in rhythms:
        b = Bar("C", (beats, 4))
        for note in rhythm:
            k = random.random()
            if k > pOfRests:
                b.place_notes(melody[melodyCount], 4 / note)
            else:
                b.place_notes(None, 4 / note)
            melodyCount += 1
        track + b
    return (track, melodyCount)


def CleanTrack(track):
    delete_clef_string = (
        " \n \override Staff.Clef.color = #white \n \override Staff.Clef.layer = #-1"
    )
    delete_time_string = " \n \override Staff.TimeSignature.color = #white \n \override Staff.TimeSignature.layer = #-1"
    track = track[0] + delete_clef_string + delete_time_string + track[1:]
    return track


def GenSingleLily(time, bars, withChrom, with16):
    track, count = NewTrack(time, bars, withChrom, with16)
    ground_lp = LilyPond.from_Track(track)
    lp = CleanTrack(ground_lp)
    main_pattern = r" [a-z]'*\d "
    occurances = re.findall(main_pattern, lp)
    count = random.randint(0, len(occurances))
    to_replace = random.sample(occurances, k=count)
    new_notes = [note[:-1] + random.choice(noteSymbols) + " " for note in to_replace]
    for pattern, replacement in zip(to_replace, new_notes):
        lp = re.sub(pattern, replacement, lp, count=1)
    return lp, ground_lp


def GenTripleLily(time, bars, withChrom, with16):
    lp, ground_lp = GenSingleLily(time, bars, withChrom, with16)
    return (
        " \\new PianoStaff \with { \override StaffGrouper.staff-staff-spacing = #'((basic-distance . 10) (padding . 10)) } << \\new Staff "
        + lp
        + " \\new Staff "
        + lp
        + " \\new Staff "
        + lp
        + " >>",
        ground_lp,
    )


def GenerateCropped(ly_string, filename, command="-fpng"):
    """Generates cropped PNG it is slightly changed version of minugs save_string_and_execute_LilyPond function"""
    ly_string = '\\version "2.10.33"\n' + ly_string
    if filename[-4] in [".pdf" or ".png"]:
        filename = filename[:-4]
    try:
        f = open(filename + ".ly", "w")
        f.write(ly_string)
        f.close()
    except:
        return False
    command = 'lilypond -dresolution=300 -dpreview %s -o "%s" "%s.ly"' % (
        command,
        filename,
        filename,
    )
    print("Executing: %s" % command)
    p = subprocess.Popen(command, shell=True).wait()
    os.remove(filename + ".ly")
    return True


def imgConvert(from_name, to_name):
    im = Image.open(from_name)
    rgb_im = im.convert("RGB")
    rgb_im.save(to_name)


def getRotationMatrixManual(rotation_angles):

    rotation_angles = [np.deg2rad(x) for x in rotation_angles]

    x_angle = rotation_angles[0]
    y_angle = rotation_angles[1]
    z_angle = rotation_angles[2]

    # X rotation
    Rx_angle = np.eye(4, 4)
    sp = np.sin(x_angle)
    cp = np.cos(x_angle)
    Rx_angle[1, 1] = cp
    Rx_angle[2, 2] = Rx_angle[1, 1]
    Rx_angle[1, 2] = -sp
    Rx_angle[2, 1] = sp

    # Y rotation
    Ry_angle = np.eye(4, 4)
    sg = np.sin(y_angle)
    cg = np.cos(y_angle)
    Ry_angle[0, 0] = cg
    Ry_angle[2, 2] = Ry_angle[0, 0]
    Ry_angle[0, 2] = sg
    Ry_angle[2, 0] = -sg

    Rz_angle = np.eye(4, 4)
    st = np.sin(z_angle)
    ct = np.cos(z_angle)
    Rz_angle[0, 0] = ct
    Rz_angle[1, 1] = Rz_angle[0, 0]
    Rz_angle[0, 1] = -st
    Rz_angle[1, 0] = st

    R = reduce(lambda x, y: np.matmul(x, y), [Rx_angle, Ry_angle, Rz_angle])

    return R


def getPoints_for_PerspectiveTranformEstimation(ptsIn, ptsOut, W, H, sidelength):

    ptsIn2D = ptsIn[0, :]
    ptsOut2D = ptsOut[0, :]
    ptsOut2Dlist = []
    ptsIn2Dlist = []

    for i in range(0, 4):
        ptsOut2Dlist.append([ptsOut2D[i, 0], ptsOut2D[i, 1]])
        ptsIn2Dlist.append([ptsIn2D[i, 0], ptsIn2D[i, 1]])

    pin = np.array(ptsIn2Dlist) + [W / 2.0, H / 2.0]
    pout = (np.array(ptsOut2Dlist) + [1.0, 1.0]) * (0.5 * sidelength)
    pin = pin.astype(np.float32)
    pout = pout.astype(np.float32)

    return pin, pout


def warpMatrix(W, H, z_angle, x_angle, y_angle, scale, fV):

    # M is to be estimated
    M = np.eye(4, 4)

    fVhalf = np.deg2rad(fV / 2.0)
    d = np.sqrt(W * W + H * H)
    sideLength = scale * d / np.cos(fVhalf)
    h = d / (2.0 * np.sin(fVhalf))
    n = h - (d / 2.0)
    f = h + (d / 2.0)

    # Translation along Z-axis by -h
    T = np.eye(4, 4)
    T[2, 3] = -h

    # Rotation matrices around x,y,z
    R = getRotationMatrixManual([x_angle, y_angle, z_angle])

    # Projection Matrix
    P = np.eye(4, 4)
    P[0, 0] = 1.0 / np.tan(fVhalf)
    P[1, 1] = P[0, 0]
    P[2, 2] = -(f + n) / (f - n)
    P[2, 3] = -(2.0 * f * n) / (f - n)
    P[3, 2] = -1.0

    F = reduce(lambda x, y: np.matmul(x, y), [P, T, R])

    ptsIn = np.array(
        [
            [
                [-W / 2.0, -H / 2.0, 0.0],
                [W / 2.0, -H / 2.0, 0.0],
                [-W / 2.0, H / 2.0, 0.0],
                [W / 2.0, H / 2.0, 0.0],
            ]
        ]
    )
    ptsOut = np.array(np.zeros((ptsIn.shape), dtype=ptsIn.dtype))
    ptsOut = cv2.perspectiveTransform(ptsIn, F)

    ptsInPt2f, ptsOutPt2f = getPoints_for_PerspectiveTranformEstimation(
        ptsIn, ptsOut, W, H, sideLength
    )
    assert ptsInPt2f.dtype == np.float32
    assert ptsOutPt2f.dtype == np.float32
    M33 = cv2.getPerspectiveTransform(ptsInPt2f, ptsOutPt2f)

    return M33, sideLength, ptsInPt2f, ptsOutPt2f


def warpImage(src, theta, phi, gamma, scale, fovy, corners=None):
    H, W, Nc = src.shape
    M, sl, ptsIn, ptsOut = warpMatrix(W, H, theta, phi, gamma, scale, fovy)
    # Compute warp matrix
    sl = int(sl)
    dst = cv2.warpPerspective(src, M, (sl, sl), borderValue=[255, 255, 255])
    # Do actual image warp
    left_right_margin = random.uniform(2, 50)
    top_bot_margin = random.uniform(2, 50)
    left_upper = [min([x[0] for x in ptsOut]), min([x[1] for x in ptsOut])]
    right_lower = [max([x[0] for x in ptsOut]), max([x[1] for x in ptsOut])]
    left_upper[0] = int(max(left_upper[0] - left_right_margin, 0))
    left_upper[1] = int(max(left_upper[1] - top_bot_margin, 0))
    right_lower[0] = int(min(right_lower[0] + left_right_margin, sl - 1))
    right_lower[1] = int(min(right_lower[1] + top_bot_margin, sl - 1))
    return dst[left_upper[1] : right_lower[1], left_upper[0] : right_lower[0]]


def randomWarpImage(src, x_range=4, y_range=8, z_range=8):
    x_angle = int(random.uniform(-x_range, x_range))
    y_angle = int(random.uniform(-y_range, y_range))
    z_angle = int(random.uniform(-z_range, z_range))
    fov = int(random.uniform(30, 50))
    warped_image = warpImage(src, x_angle, y_angle, z_angle, 1, fov)
    return warped_image[:, :, :]


def handle_single_track():
    beats = random.choices([3, 4], weights=[0.25, 0.75], k=1)[0]
    count = random.choices([1, 2, 3, 4, 5], weights=[1, 1, 1, 1, 1], k=1)[0]
    image_track, ground_track = GenSingleLily(
        beats, count, withChrom=False, with16=True
    )
    GenerateCropped(image_track, "temp_to_split")
    src = cv2.imread("temp_to_split.preview.png")
    src = src[..., ::-1]  # BGR to RGB
    src = randomWarpImage(src)
    # im = Image.fromarray(src)
    # plt.imshow(src)
    H, W, Nc = src.shape
    src = src[:, 170:, :]
    return ground_track, image_track, src


def handle_multi_track():
    beats = random.choices([3, 4], weights=[0.25, 0.75], k=1)[0]
    count = random.choices([1, 2, 3, 4, 5], weights=[1, 1, 1, 1, 1], k=1)[0]
    image_track, ground_track = GenTripleLily(
        beats, count, withChrom=False, with16=True
    )
    GenerateCropped(image_track, "temp_to_split")
    src = cv2.imread("temp_to_split.preview.png")
    # src = src[..., ::-1]  # BGR to RGB
    src = randomWarpImage(src)
    H, W, Nc = src.shape
    r = random.random()
    if r < 0.33:
        src = src[: H // 3, :, :]
    elif r < 0.66:
        src = src[H // 3 : 2 * H // 3, :, :]
    else:
        src = src[2 * H // 3 :, :, :]
    src = src[:, 170:, :]
    return ground_track, image_track, src


def randomBlurImage(src, kernel=7, sigma_min=0, sigma_max=3):
    sigma = random.randint(sigma_min, sigma_max)
    gaussian_blur = cv2.GaussianBlur(src, (kernel, kernel), sigmaX=sigma)
    return gaussian_blur


def randomNoise(src, mean_min=3, mean_max=5, var_min=1, var_max=9):
    m = random.randint(mean_min, mean_max) / 10
    v = random.randint(var_min, var_max) / 100
    noisy_image = skimage.util.random_noise(src, mean=m, var=v)
    noisy_image = np.clip(noisy_image * 255, 0, 255).astype(np.uint8)
    return noisy_image


def addRandomDist(src):
    if random.random() < 0.33:
        src = randomBlurImage(src)
    if random.random() < 0.33:
        src = randomNoise(src)
    return src


def adaptLilyForModel(lp):
    new_lilypond = lp.replace("{ {", "").replace("} {", "|").replace("} }", "|")
    return new_lilypond


def GenerateRandomPhoto(name, data_dir_path="./Data"):
    if random.random() <= 0.25:
        ground_track, image_track, src = handle_single_track()
    else:
        ground_track, image_track, src = handle_multi_track()
    src = addRandomDist(src)
    grayImage = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    post_bin = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    src = cv2.cvtColor(post_bin, cv2.COLOR_GRAY2RGB)
    ground_track = (
        ground_track.replace("{ {", "").replace("} {", "|").replace("} }", "|")
    )
    ground_track = ground_track.replace("\\time 3/4 ", "")
    im = Image.fromarray(src)
    os.mkdir(f"{data_dir_path}/{name}")
    im.save(f"{data_dir_path}/{name}/{name}.jpg")
    with open(f"{data_dir_path}/{name}/{name}.txt", "w") as text_file:
        text_file.write(ground_track)


def main():
    for i in range(0, 100000):
        GenerateRandomPhoto(str(i))


if __name__ == "__main__":
    main()
