import sys
import cv2
import numpy as np

global l, vidWriter, img, width_x, height_y,step_size, num_rays, cost_map, node_cnt
sys.setrecursionlimit(10 ** 9)
width_x = 300
height_y = 200
run_2_goal = 200000
running_cost_2_goal = np.inf

def sign(point1, point2, point3):
    return (point1[0] - point3[0]) * (point2[1] - point3[1]) - (point2[0] - point3[0]) * (point1[1] - point3[1])

def draw_map(clearance):  # draws the map with the clearance
    global img

    img = np.zeros([height_y,width_x,3],dtype=np.uint8)

    cv2.circle(img, (225, 150), 25, (255, 0, 0), clearance * 2)  # circle clearance
    cv2.circle(img, (225, 150), 25, (255, 255, 255), 1)  # circle outline

    cv2.ellipse(img=img,  # ellipse clearance
                center=(150, 100),
                axes=(40, 20),
                angle=0,
                startAngle=0,
                endAngle=360,
                color=(255, 0, 0),
                thickness=clearance * 2)

    cv2.ellipse(img=img,  # ellipse outline
                center=(150,100),
                axes=(40,20),
                angle=0,
                startAngle=0,
                endAngle=360,
                color=(255,255,255),
                thickness=1)

    point1 = (95, 30)  # points used to produce the rectangle
    point2 = (int(95 + 10 * np.sin(np.deg2rad(30))), int(30 + 10 * np.cos(np.deg2rad(30))))
    point3 = (int(point2[0] - 75 * np.cos(np.deg2rad(30))), int(point2[1] + 75 * np.sin(np.deg2rad(30))))
    point4 = (int(95 - 75 * np.cos(np.deg2rad(30))), int(30 + 75 * np.sin(np.deg2rad(30))))

    cv2.line(img, point1, point2, (255, 0, 0), clearance*2)
    cv2.line(img, point2, point3, (255, 0, 0), clearance*2)
    cv2.line(img, point3, point4, (255, 0, 0), clearance*2)
    cv2.line(img, point4, point1, (255, 0, 0), clearance*2)

    cv2.line(img, point1, point2, (255, 255, 255), 1)
    cv2.line(img, point2, point3, (255, 255, 255), 1)
    cv2.line(img, point3, point4, (255, 255, 255), 1)
    cv2.line(img, point4, point1, (255, 255, 255), 1)

    cv2.line(img, (20, 120), (25, 185), (255, 0, 0), clearance*2)  # points used to create concave shape
    cv2.line(img, (25, 185), (75, 185), (255, 0, 0), clearance*2)
    cv2.line(img, (75, 185), (100, 150), (255, 0, 0), clearance*2)
    cv2.line(img, (100, 150), (75, 120), (255, 0, 0), clearance*2)
    cv2.line(img, (75, 120), (50, 150), (255, 0, 0), clearance*2)
    cv2.line(img, (50, 150), (20, 120), (255, 0, 0), clearance*2)
    cv2.line(img, (225, 10), (250, 25), (255, 0, 0), clearance*2)
    cv2.line(img, (250, 25), (225, 40), (255, 0, 0), clearance*2)
    cv2.line(img, (225, 10), (200, 25), (255, 0, 0), clearance*2)
    cv2.line(img, (200, 25), (225, 40), (255, 0, 0), clearance*2)

    cv2.line(img, (20,120), (25,185), (255, 255, 255), 1)
    cv2.line(img, (25,185), (75,185), (255, 255, 255), 1)
    cv2.line(img, (75,185), (100,150), (255, 255, 255), 1)
    cv2.line(img, (100,150), (75,120), (255, 255, 255), 1)
    cv2.line(img, (75,120), (50,150), (255, 255, 255), 1)
    cv2.line(img, (50,150), (20,120), (255, 255, 255), 1)
    cv2.line(img, (225,10), (250,25), (255, 255, 255), 1)
    cv2.line(img, (250,25), (225,40), (255, 255, 255), 1)
    cv2.line(img, (225, 10), (200, 25), (255, 255, 255), 1)
    cv2.line(img, (200, 25), (225, 40), (255, 255, 255), 1)

    cv2.circle(img, tuple(end_pt), int(1.5), (0, 0, 255), 1)
    cv2.circle(img, tuple(start_pt), int(1.5), (0, 255, 0), 1)

    cv2.imshow('image', cv2.flip(img,0))

    cv2.waitKey()

def is_in_circle_obstacle(point, xpos, ypos, radius,clearance):  # checks if point is in circle
    if np.sqrt(np.square(ypos - point[1]) + np.square(xpos - point[0])) <= radius+clearance:
        return True

def is_in_oval_obstacle(point, xpos, ypos, radius_x, radius_y,clearance):  # checks if point is in an oval
    first_oval_term = np.square(point[0] - xpos) / np.square(radius_x+clearance)
    second_oval_term = np.square(point[1] - ypos) / np.square(radius_y+clearance)
    if first_oval_term + second_oval_term <= 1:
        return True

def is_in_triangle_obstacle(point, three_points):  # checks if a point is in a triangle
    v1 = three_points[0]
    v2 = three_points[1]
    v3 = three_points[2]
    pt = point
    d1 = sign(pt, v1, v2)
    d2 = sign(pt, v2, v3)
    d3 = sign(pt, v3, v1)
    neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    if not (neg and pos):
        return True

def is_in_slot(point,point1,point2,clearance):  # checks if a point is in a slot or an offset line with clearance

    if is_in_circle_obstacle(point,point1[0],point1[1],0,clearance):
        return True
    if is_in_circle_obstacle(point, point2[0], point2[1], 0, clearance):
        return True


    perp_angle1 = np.arctan2(point1[1]-point2[1],point1[0]-point2[0])+np.pi/2
    perp_angle2 = np.arctan2(point1[1]-point2[1],point1[0]-point2[0])+(3.0/2.0)*np.pi

    point11 = [point1[0]+clearance*np.cos(perp_angle1),point1[1]+clearance*np.sin(perp_angle1)]
    point12 = [point1[0]+clearance*np.cos(perp_angle2),point1[1]+clearance*np.sin(perp_angle2)]
    point21 = [point2[0]+clearance*np.cos(perp_angle1),point2[1]+clearance*np.sin(perp_angle1)]
    point22 = [point2[0]+clearance*np.cos(perp_angle2),point2[1]+clearance*np.sin(perp_angle2)]

    if is_in_triangle_obstacle(point,[point11,point12,point21]):
        return True
    if is_in_triangle_obstacle(point,[point11,point12,point22]):
        return True
    if is_in_triangle_obstacle(point,[point22,point21,point11]):
        return True
    if is_in_triangle_obstacle(point,[point22,point21,point12]):
        return True

def is_not_on_map(point):  # checks to see if the point is actually on the map
    if int(round(point[0])) >= 0 and int(round(point[0])) >= cost_map.shape[0]-1:
        return True
    if int(round(point[1])) >= 0 and int(round(point[1])) >= cost_map.shape[1]-1:
        return True

def is_valid_point_on_map(point,clearance):  # Checks is a point against all map obstacles

    if point[0] < 0:
        return False
    if point[1] < 0:
        return False

    if is_not_on_map(point):
        return False
    if is_in_circle_obstacle(point,225,150,25,clearance):
        return False
    if is_in_oval_obstacle(point,150,100,40,20,clearance):
        return False
    if is_in_triangle_obstacle(point,([20, 120], [25, 185], [50, 150])):
        return False
    if is_in_triangle_obstacle(point,([50, 150], [25, 185], [75, 185]) ):
        return False
    if is_in_triangle_obstacle(point,([50, 150], [75, 185], [100, 150])):
        return False
    if is_in_triangle_obstacle(point,([50, 150], [100, 150], [75, 120]) ):
        return False
    if is_in_triangle_obstacle(point, ([25, 185], [50, 150], [75, 185])):
        return False
    if is_in_triangle_obstacle(point,([75, 185], [100, 150], [50, 150]) ):
        return False
    if is_in_triangle_obstacle(point,([50, 150], [100, 150], [75, 120])):
        return False
    if is_in_triangle_obstacle(point,([225, 10], [225, 40], [250, 25]) ):
        return False
    if is_in_triangle_obstacle(point,([225, 10], [225, 40], [200, 25]) ):
        return False
    point1 = [95, 30]
    point2 = [95 + 10 * np.sin(np.deg2rad(30)), 30 + 10 * np.cos(np.deg2rad(30))]
    point3 = [point2[0] - 75 * np.cos(np.deg2rad(30)), point2[1] + 75 * np.sin(np.deg2rad(30))]
    point4 = [95 - 75 * np.cos(np.deg2rad(30)), 30 + 75 * np.sin(np.deg2rad(30))]
    if is_in_triangle_obstacle(point, (point3, point1, point2)):
        return False
    if is_in_triangle_obstacle(point, (point4, point3, point1)):
        return False
    if is_in_slot(point,point1,point2,clearance):
        return False
    if is_in_slot(point,point2,point3,clearance):
        return False
    if is_in_slot(point, point3, point4, clearance):
        return False
    if is_in_slot(point, point4, point1, clearance):
        return False
    if is_in_slot(point, (20, 120), (25, 185), clearance):
        return False
    if is_in_slot(point, (25, 185), (75, 185),clearance):
        return False
    if is_in_slot(point, (75, 185), (100, 150),clearance):
        return False
    if is_in_slot(point, (100, 150), (75, 120),clearance):
        return False
    if is_in_slot(point, (75, 120), (50, 150),clearance):
        return False
    if is_in_slot(point, (50, 150), (20, 120),clearance):
        return False
    if is_in_slot(point, (225, 10), (250, 25),clearance):
        return False
    if is_in_slot(point, (250, 25), (225, 40),clearance):
        return False
    if is_in_slot(point, (225, 10), (200, 25),clearance):
        return False
    if is_in_slot(point, (200, 25), (225, 40),clearance):
        return False
    return True

def cost_2_goal(point): # gives the cost to the goal from a given point
    if point[0] - end_pt[0] == 0:
        cost = abs(point[1] - end_pt[1])
    elif point[1] - end_pt[1] == 0:
        cost = abs(point[0] - end_pt[0])
    else:
        cost = np.sqrt(np.square(point[0] - end_pt[0]) + np.square(point[1] - end_pt[1]))
    return cost

def is_goal(point):  # checks if a point is also the goal
    if cost_2_goal(point) <= goal_threshold: # this fix
        return True

class node:  # makes nodes that will be used to save and explore child states

    def __init__(self, location, orientation, cost2come,cost2goal,value,parent):
        global node_cnt
        self.loc = location
        self.orientation = orientation
        self.cost2come = cost2come
        self.cost2goal = cost2goal
        self.parent = parent
        self.value = value
        self.value = cost2come + cost2goal
        self.counter = node_cnt
        node_cnt += 1


def find_path(curr_node):  # Draws the path to the goal on the map
    while(curr_node!=None):
        try:
            curr_node = curr_node.parent
            current_point = (int(curr_node.loc[0]),int(curr_node.loc[1]))
            parent_node = curr_node.parent
            parent_point = (int(parent_node.loc[0]),int(parent_node.loc[1]))
            cv2.line(img,current_point,parent_point,(225,50,50),1)
        except:
            pass
    for i in range(1000):
        vidWriter.write(cv2.flip(img, 0))
    vidWriter.release()

def find_children(curr_node):  # finds the child states from a given node

    curr_location = curr_node.loc
    curr_orientation = curr_node.orientation
    curr_cost2come = curr_node.cost2come
    child_orientations = action_angles
    allowable_childern = []
    for action in child_orientations:
        child_point = [curr_location[0] + step_size*np.cos(np.deg2rad(action)),
        curr_location[1] + step_size*np.sin(np.deg2rad(action))]

        if is_valid_point_on_map(child_point,total_clear):
            childcost2come = curr_cost2come+step_size
            cost2goal = cost_2_goal(child_point)
            total_cost = childcost2come+cost2goal

            if cost_map[int(round(child_point[0])), int(round(child_point[1]))] > childcost2come:
                cost_map[int(round(child_point[0])), int(round(child_point[1]))] = childcost2come

                child_node = node(location=child_point,
                                      orientation=action+curr_orientation,
                                      cost2come=childcost2come,
                                      value = total_cost,
                                      cost2goal=cost_2_goal(child_point),
                                      parent=curr_node)
                allowable_childern.append((child_node.value, child_node.counter, child_node))
                img[int(round(child_point[1])), int(round(child_point[0])), 0:3] = [255, 255, 255]
                vidWriter.write(cv2.flip(img, 0))
                cv2.line(img,(int(child_point[0]),int(child_point[1])),(int(curr_location[0]),int(curr_location[1])),(0,255,255),thickness=1)


    children_list = allowable_childern
    return children_list  # list of all children with a lesser cost for the current node

def solver(l):  # finds the lowest cost in unexplored states, deletes the state, and adds child states

    goal_not_found = 1
    while goal_not_found:

        curr_node_ary = min(l)
        curr_node = curr_node_ary[2]
        l.remove(min(l))
        if (is_goal(curr_node.loc)):
            find_path(curr_node)  # find the path to the start node by tracking the node's parent
            print("here we found a goal")
            goal_not_found = 0
        children_list = find_children(curr_node)  # a function to find possible children and update cost
        l = l + children_list  # adding possible children to the list
    return 1

if __name__ == "__main__":
    global start_pt
    global end_pt
    global vidWriter
    global path
    global node_cnt
    global goal_threshold
    node_cnt = 0

    cost_map = np.inf*np.ones((width_x,height_y))
    vidWriter = cv2.VideoWriter("A_start.avi", cv2.VideoWriter_fourcc('X','V','I','D'), 500, (width_x, height_y))
    bot_r = int(input("Enter robot radius: "))
    clear_r = int(input("Enter the clearance: "))
    total_clear = bot_r + clear_r



    valid_points = False
    running_cost_2_goal = np.inf

    while valid_points == False:
        start_state = (input("Enter start point in form # # # for X Y Orientation: "))
        start_state = [int(start_state.split()[0]), int(start_state.split()[1]),int(start_state.split()[1])]
        start_pt = [int(round(start_state[0])), int(round(start_state[1]))]
        start_orientation = start_state[2]


        end_state = (input("Enter end point in form # # for X Y: "))
        end_state = [int(end_state.split()[0]), int(end_state.split()[1])]
        end_pt = [int(round(end_state[0])), int(round(end_state[1]))]


        if is_valid_point_on_map(start_pt,total_clear) == False or is_valid_point_on_map(end_pt,total_clear) == False:
            # check if either the start or end node an obstacle
            print("Enter valid points... ")
            continue
        else:
            valid_points = True
    draw_map(total_clear)
    step_size = float(input('Enter step size: '))

    theta_threshold = float(input('Enter angle between rays, must be a factor of 360: '))
    goal_threshold = float(input('Enter node distance from goal that will register as a found goal: '))
    action_angles = np.arange(theta_threshold,360+theta_threshold,theta_threshold)

    start_node = node(location=start_pt,
                      orientation=start_orientation,
                      value=cost_2_goal(start_pt),
                      cost2come=0,
                      cost2goal=cost_2_goal(start_pt),
                      parent=None)
    cost_map[start_pt[0],start_pt[1]] = 0
    global l

    l = [(start_node.value, start_node.counter, start_node)]

    print("Running..")

    flag = solver(l)
    if flag == 1:
        print("Path found. Please watch the video generated.")
    else:
        print("Solution not found... ")


