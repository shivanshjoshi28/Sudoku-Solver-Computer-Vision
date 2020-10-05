import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from solve_puzzle import solve, check_if_solvable, verify
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import urllib.request
####SETTING SPACE ####

debug = 1  # use debug as 0 if we don't want ot debug our code
model = load_model('model.hdf5')
URL="http://100.81.10.218:8080/shot.jpg"

######################


def process(img):
    kernel = np.ones((2, 2))
    if (len(img.shape) == 3):
        greyscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        greyscale_img = img
    Blurred_img = cv2.GaussianBlur(greyscale_img, (9, 9), 0)
    thresh_img = cv2.adaptiveThreshold(Blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # this will convert all no . above 11 to 255 and below to 0 as it is a binary thresh.
    inverted_img = cv2.bitwise_not(thresh_img,
                                   0)  # it will reverse the binary thresh..... I use this because it is noticed that INV BINARY THRESH is not successfull
    morph_img = cv2.morphologyEx(inverted_img, cv2.MORPH_OPEN, kernel)  # applying morphological opening
    #     cv2.imshow("morphed pic",morph_img)
    dilated_img = cv2.dilate(morph_img, kernel,
                             iterations=1)  # applying dilation of the image which will remove all the opening around the number and also darken the line
    return dilated_img


def get_corners(img):
    contours, hire = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    largest_contour = np.squeeze(contours[0])
    #     maxcont=contours[0]
    #     maxarea=cv2.contourArea(contours[0])
    #     for i in contours:
    #         area = cv2.contourArea(i)
    #         epsilon_each_contour = 0.01 * cv2.arcLength(i, True)
    #         poly_approx = cv2.approxPolyDP(i, epsilon_each_contour, True)
    #         if (len(poly_approx) == 4 and area > maxarea):
    #             print("hlo")
    #             maxarea = area
    #             maxcont = poly_approx
    #     largest_contour=np.squeeze(maxcont)

    sums = [sum(i) for i in largest_contour]
    difference = [i[0] - i[1] for i in largest_contour]
    top_left = np.argmin(sums)
    top_right = np.argmax(difference)
    bottom_right = np.argmax(sums)
    bottom_left = np.argmin(difference)

    corners = [
        largest_contour[top_left],
        largest_contour[top_right],
        largest_contour[bottom_right],
        largest_contour[bottom_left]
    ]
    return corners, largest_contour


def transform(pts, img):
    pts = np.float32(pts)
    # print("pts", pts)
    top_l, top_r, bot_r, bot_l = pts[0], pts[1], pts[2], pts[3]

    def pythagoras(pt1, pt2):
        return np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)

    width = int(max(pythagoras(bot_r, bot_l), pythagoras(top_r, top_l)))
    height = int(max(pythagoras(top_r, bot_r), pythagoras(top_l, bot_l)))
    square = max(width, height) // 9 * 9  # Making the image dimensions divisible by 9

    dim = np.array(
        ([0, 0], [square - 1, 0], [square - 1, square - 1], [0, square - 1]), dtype="float32"
    )
    # print("dim", dim)
    matrix = cv2.getPerspectiveTransform(pts, dim)
    warped = cv2.warpPerspective(img, matrix, (square, square))
    return warped


def get_grid_lines(img, length=12):
    horizontal = np.copy(img)
    cols = horizontal.shape[1]
    horizontal_size = cols // length
    kernel = np.ones((1, horizontal_size))

    horizontal = cv2.erode(horizontal, kernel)
    horizontal = cv2.dilate(horizontal, kernel)

    vertical = np.copy(img)
    rows = vertical.shape[0]
    vertical_size = rows // length
    kernel = np.ones((vertical_size, 1))

    vertical = cv2.erode(vertical, kernel)
    vertical = cv2.dilate(vertical, kernel)
    return vertical, horizontal


####### Another method to form mask ########
# (We don't use this method bcs it might fail in some case when the line are not that clearer........ that time hough lines transform is better)
def create_grid_mask1(vertical, horizontal):
    grid = cv2.add(horizontal, vertical)
    grid = cv2.adaptiveThreshold(
        grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 235, 2
    )
    grid = cv2.dilate(grid, np.ones((3, 3)), iterations=2)

    grid = cv2.cvtColor(grid, cv2.COLOR_GRAY2RGB)
    print("grid shape", grid.shape)
    print(grid[0][0][0], grid[0][0][1], grid[0][0][2])  # we notice this as white
    grid_red = grid.copy()

    for i, a in enumerate(grid_red):
        for j, v in enumerate(a):
            grid_red[i][j][0] = 0
            grid_red[i][j][1] = 0
    cv2.imshow("Grid_red", grid_red)
    cv2.imshow("Mask through 1st function", cv2.bitwise_not(grid))


#############################################


def create_grid_mask(vertical, horizontal):
    grid = cv2.add(horizontal, vertical)
    grid = cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 235, 2)
    grid = cv2.dilate(grid, np.ones((3, 3)), iterations=2)
    pts = cv2.HoughLines(grid, .3, np.pi / 90, 200)

    def draw_lines(im, pts):
        im = np.copy(im)
        pts = np.squeeze(pts)
        for r, theta in pts:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * r
            y0 = b * r
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv2.line(im, (x1, y1), (x2, y2), (255, 255, 255), 2)
        return im

    lines = draw_lines(grid, pts)
    mask = cv2.bitwise_not(lines)
    return mask


def extract_digits(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img.shape[0] * img.shape[1]
    # Reversing contours list to loop with y coord ascending, and removing small bits of noise
    contours_denoise = [i for i in contours[::-1] if cv2.contourArea(i) > img_area * .0005]
    _, y_compare, _, _ = cv2.boundingRect(contours_denoise[0])
    digits = []
    row = []

    for i in contours_denoise:
        x, y, w, h = cv2.boundingRect(i)
        cropped = img[y:y + h, x:x + w]
        if y - y_compare > img.shape[1] // 40:
            row = [i[0] for i in sorted(row, key=lambda x: x[1])]
            for j in row:
                digits.append(j)
            row = []
        row.append((cropped, x))
        y_compare = y
    # Last loop doesn't add row
    row = [i[0] for i in sorted(row, key=lambda x: x[1])]
    for i in row:
        digits.append(i)

    return digits


def add_border(img_arr):
    digits = []
    for i in img_arr:
        crop_h, crop_w = i.shape[:2]
        try:
            pad_h = int(crop_h / 1.75)
            pad_w = (crop_h - crop_w) + pad_h
            pad_h //= 2
            pad_w //= 2
            border = cv2.copyMakeBorder(i, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            digits.append(border)
        except cv2.error:
            continue
    dims = (digits[0].shape[0],) * 2
    digits_square = [cv2.resize(i, dims, interpolation=cv2.INTER_NEAREST) for i in digits]
    return digits_square


def subdivide(img, divisions=9):
    height, _ = img.shape[:2]
    box = height // divisions
    if len(img.shape) > 2:
        subdivided = img.reshape(height // box, box, -1, box, 3).swapaxes(1, 2).reshape(-1, box, box, 3)
    else:
        subdivided = img.reshape(height // box, box, -1, box).swapaxes(1, 2).reshape(-1, box, box)
    return [i for i in subdivided]


def add_zeros(sorted_arr, subd_arr):
    h, w = sorted_arr[0].shape
    puzzle_template = np.zeros((81, h, w), dtype=np.uint8)
    sorted_arr_idx = 0
    for i, j in enumerate(subd_arr):
        if np.sum(j) < 9000:
            zero = np.zeros((h, w), dtype=np.uint8)
            puzzle_template[i] = zero
        else:
            puzzle_template[i] = sorted_arr[sorted_arr_idx]
            sorted_arr_idx += 1
    return puzzle_template


def img_to_array(img_arr, img_dims):
    predictions = []
    for i in img_arr:
        resized = cv2.resize(i, (img_dims, img_dims), interpolation=cv2.INTER_LANCZOS4)
        if np.sum(resized) == 0:
            predictions.append(0)
            continue
        array = np.array([resized])
        reshaped = array.reshape(array.shape[0], img_dims, img_dims, 1)
        flt = reshaped.astype('float32')
        flt /= 255
        prediction = model.predict_classes(flt)
        predictions.append(prediction[0] + 1)  # OCR predicts from 0-8, changing it to 1-9
    puzzle = np.array(predictions).reshape((9, 9))
    return puzzle


def put_solution(img_arr, soln_arr, unsolved_arr):
    solutions = np.array(soln_arr).reshape(81)
    unsolveds = np.array(unsolved_arr).reshape(81)
    paired = list((zip(solutions, unsolveds, img_arr)))
    img_solved = []
    for solution, unsolved, img in paired:
        if solution == unsolved:
            img_solved.append(img)
            continue
        img_h, img_w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        posx = img_w // 4 - img_w // 11
        posy = 2 * img_h // 3 + img_h // 11
        cv2.putText(img_rgb, str(solution), (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        #         pil_img = Image.fromarray(img_rgb)
        #         draw = ImageDraw.Draw(pil_img)
        #         fnt = ImageFont.truetype(font_path, img_h)
        #         font_w, font_h = draw.textsize(str(solution), font=fnt)
        #         draw.text(((img_w - font_w) / 2, (img_h - font_h) / 2 - img_h // 10), str(solution),
        #                   fill=(font_color if len(img.shape) > 2 else 0), font=fnt)
        cv2_img = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
        img_solved.append(cv2_img)
    return img_solved


def stitch_img(img_arr, img_dims):
    result = Image.new('RGB' if len(img_arr[0].shape) > 2 else 'L', img_dims)
    box = [0, 0]
    for img in img_arr:
        pil_img = Image.fromarray(img)
        result.paste(pil_img, tuple(box))
        if box[0] + img.shape[1] >= result.size[1]:
            box[0] = 0
            box[1] += img.shape[0]
        else:
            box[0] += img.shape[1]
    return np.array(result)


def inverse_perspective(img, dst_img, pts):
    pts_source = np.array([[0, 0], [img.shape[1] - 1, 0], [img.shape[1] - 1, img.shape[0] - 1], [0, img.shape[0] - 1]],
                          dtype='float32')
    h, status = cv2.findHomography(pts_source, pts)
    warped = cv2.warpPerspective(img, h, (dst_img.shape[1], dst_img.shape[0]))
    cv2.fillConvexPoly(dst_img, np.ceil(pts).astype(int), 0, 16)
    dst_img = dst_img + warped
    return dst_img

def solve_photo(img_path = 'home3.jpeg'):
    while True:
        img = cv2.imread(img_path)
        processed_image = process(img)
        corners, contour = get_corners(processed_image)
        warped = transform(corners, processed_image)
        vertical_lines, horizontal_lines = get_grid_lines(warped)
        #     print("v.s",vertical_lines.shape)
        mask = create_grid_mask(vertical_lines, horizontal_lines)
        #     print("Mask shape",mask.shape)
        #     print("warped shape",warped.shape)
        numbers = cv2.bitwise_and(warped, mask)
        cv2.imshow("number", numbers)
        digits_sorted_array = extract_digits(numbers)  # it store the number rowise in ascending order
        digits_border_array = add_border(
            digits_sorted_array)  # it is the array of the digits sorted with increased border.
        digits_subd_array = subdivide(numbers)  # it store the each number box (inner box)

        try:
            digits_with_zeros = add_zeros(digits_border_array,
                                          digits_subd_array)  # it will make the box black i.e 0 full to the box , that doesn't have the no.
            # print(digits_with_zeros.shape)
        except IndexError:
            sys.stderr.write('ERROR: Image too warped')
            sys.exit()

        try:
            puzzle = img_to_array(digits_with_zeros, 32)  # as we have trained the model on the image of 32 * 32 pixel
            print(puzzle)
        except AttributeError:
            sys.stderr.write('ERROR: OCR predictions failed')
            sys.exit()

        solved = solve(puzzle.copy().tolist())  # Solve function modifies original puzzle var
        if not solved:
            raise ValueError('ERROR: Puzzle not solvable')

        warped_img = transform(corners, img)
        subd = subdivide(warped_img)
        subd_soln = put_solution(subd, solved, puzzle)
        #     for i in subd_soln:   # this is only for debugging purpose
        #         plt.imshow(i)
        #         plt.show()
        warped_soln = stitch_img(subd_soln, (warped_img.shape[0], warped_img.shape[1]))
        plt.imshow(warped_soln)
        plt.show()
        warped_inverse = inverse_perspective(warped_soln, img.copy(), np.array(corners))
        cv2.imshow("Solved Soduku ", warped_inverse)

        if (debug):
            cv2.imshow("O", processed_image)
            for i in corners:
                i = tuple(i)
                cv2.circle(img, i, 4, (0, 0, 255), 2)  # this is to mark circle at the corners
            cv2.drawContours(img, [contour], 0, (0, 255, 0),
                             2)  # this is to make boundry outline the max area square found in corner function
            create_grid_mask1(vertical_lines, horizontal_lines)

            cv2.imshow("Warped image", warped)
            cv2.imshow("Horizontal", horizontal_lines)
            cv2.imshow("Vertical", vertical_lines)
            cv2.imshow("Original image", img)
            cv2.imshow("Numbers image", numbers)

        if (cv2.waitKey(0) == 13):
            break

    cv2.destroyAllWindows()



# solve_photo()



def solve_webcam(debug=True):

    cap = cv2.VideoCapture(0)
    stored_soln = []
    stored_puzzle = []
    # Creating placeholder grid to match against until one is taken from the sudoku puzzle
    cells = [np.pad(np.ones((7, 7), np.uint8) * 255, (1, 1), 'constant', constant_values=(0, 0)) for _ in range(81)]
    grid = stitch_img(cells, (81, 81))
    while True:
        # cv2.imshow("grid",grid)
        # print("opening camera")
        # imgResp = urllib.request.urlopen(URL)
        # imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        # imgOriginal = cv2.imdecode(imgNp, -1)
        # img = np.asarray(imgOriginal)
        # img=cv2.resize(img,(640,480))
        ret, frame = cap.read()
        img=frame
        # img = resize_keep_aspect(frame)
        try:
            processed = process(img)
            corners,contour = get_corners(processed)
            for i in corners:
                i = tuple(i)
                cv2.circle(img, i, 4, (0, 0, 255), 2)  # this is to mark circle at the corners
            # cv2.drawContours(img, [contour], 0, (0, 255, 0),
            #                  2)  # this is to make boundry outline the max area square found in corner function
            warped = transform(corners, processed)
            vertical_lines, horizontal_lines = get_grid_lines(warped)
            mask = create_grid_mask(vertical_lines, horizontal_lines)
            # cv2.imshow("Mask",mask)
            # cv2.imshow("Grid",grid)

            # Checks to see if the mask matches a grid-like structure
            template = cv2.resize(grid, (warped.shape[0],) * 2, interpolation=cv2.INTER_NEAREST)
            res = cv2.matchTemplate(mask, template, cv2.TM_CCORR_NORMED)
            threshold = .55
            loc = np.array(np.where(res >= threshold))
            if loc.size == 0:
                raise ValueError('Grid template not matched')

            if stored_soln and stored_puzzle:
                warped_img = transform(corners, img)
                subd = subdivide(warped_img)
                subd_soln = put_solution(subd, stored_soln, stored_puzzle)
                warped_soln = stitch_img(subd_soln, (warped_img.shape[0], warped_img.shape[1]))
                warped_inverse = inverse_perspective(warped_soln, img, np.array(corners))
                cv2.imshow('frame', warped_inverse)
                # print("Continuing to next frame from image")
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                print("here doing continue")
                continue





            numbers = cv2.bitwise_and(warped, mask)
            cv2.imshow("numbers",numbers)
            digits_sorted = extract_digits(numbers)
            digits_border = add_border(digits_sorted)
            digits_subd = subdivide(numbers)
            digits_with_zeros = add_zeros(digits_border, digits_subd)
            puzzle = img_to_array(digits_with_zeros, 32)
            print(puzzle)

            if np.sum(puzzle) == 0:
                raise ValueError('False positive')

            if not check_if_solvable(puzzle):
                raise ValueError('OCR Prediction wrong')

            solved = solve(puzzle.copy().tolist())
            if not solved:
                raise ValueError('Puzzle not solvable')

            if verify(solved):
                stored_puzzle = puzzle.tolist()
                stored_soln = solved
                # grid = mask

            warped_img = transform(corners, img)
            subd = subdivide(warped_img)
            subd_soln = put_solution(subd, solved, puzzle)
            warped_soln = stitch_img(subd_soln, (warped_img.shape[0], warped_img.shape[1]))
            cv2.imshow("warped sol",warped_soln)
            warped_inverse = inverse_perspective(warped_soln, img, np.array(corners))
            cv2.imshow('frame', warped_inverse)
            print("hello sir ")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            cv2.imshow('frame', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if debug:
                print(e)
            print("Its error continue")
            continue
solve_webcam()