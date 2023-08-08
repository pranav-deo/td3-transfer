from sentence_transformers import SentenceTransformer
import numpy as np
import itertools

model = SentenceTransformer('all-MiniLM-L6-v2')

task_descriptions = {}
# task_descriptions['fetch_reach_long'] = "The task in the environment is for a manipulator to move the end effector to a randomly selected position in the robot’s workspace. The robot is a 7-DoF Fetch Mobile Manipulator with a two-fingered parallel gripper. The robot is controlled by small displacements of the gripper in Cartesian coordinates. The task is also continuing which means that the robot has to maintain the end effector’s position for an indefinite period of time."
# task_descriptions['fetch_push_long'] = "The task in the environment is for a manipulator to move a block to a target position on top of a table by pushing with its gripper. The robot is a 7-DoF Fetch Mobile Manipulator with a two-fingered parallel gripper. The robot is controlled by small displacements of the gripper in Cartesian coordinates. The gripper is locked in a closed configuration in order to perform the push task. The task is also continuing which means that the robot has to maintain the block in the target position for an indefinite period of time."
# task_descriptions['fetch_slide_long'] = "The task in the environment is for a manipulator hit a puck in order to reach a target position on top of a long and slippery table. The table has a low friction coefficient in order to make it slippery for the puck to slide and be able to reach the target position which is outside of the robot’s workspace. The robot is a 7-DoF Fetch Mobile Manipulator with a two-fingered parallel gripper. The robot is controlled by small displacements of the gripper in Cartesian coordinates. The gripper is locked in a closed configuration since the puck doesn’t need to be graspped. The task is also continuing which means that the robot has to maintain the puck in the target position for an indefinite period of time."

task_descriptions['fetch_reach'] = "Move the end effector to a randomly selected position in the robot’s workspace."
task_descriptions['fetch_push'] = "Move a block to a target position on top of a table by pushing with its gripper."
task_descriptions['fetch_slide'] = "Hit a puck such that it reaches a target position on top of a long and slippery table."

# sentence_embeddings = model.encode(list(task_descriptions.values()))
sentence_embeddings = model.encode(list(task_descriptions.values())[0])

print(sentence_embeddings)

# for i, j in itertools.combinations(zip(task_descriptions.keys(), sentence_embeddings), 2):
#     print(f"Distances {i[0]}, {j[0]}: {np.dot(i[1],j[1])}")

# for sentence, embedding in zip(task_descriptions.values(), sentence_embeddings):
#     print("Sentence:", sentence)
#     print("Embedding:", embedding.shape)
#     print("")