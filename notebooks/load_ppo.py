import matplotlib.pyplot as plt
from matplotlib import animation
def save_frames_as_gif(frames, filename):
    
    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1]/100, frames[0].shape[0]/100), dpi=300)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(filename, writer='pillow', fps=60)

actor_pre=ActorSoftmax(4,2)
layer_state_dict = torch.load("models/actor.pt",map_location=torch.device('cpu'))
actor_pre.load_state_dict(layer_state_dict)

env=gym.make('CartPole-v1')

state=env.reset()
frames = []
for i in range(250):
    #print(env.render(mode="rgb_array"))
    frames.append(env.render(mode="rgb_array"))
    state = torch.tensor(state, device="cpu", dtype=torch.float32).unsqueeze(dim=0)
    action =actor_pre(state)
    action=np.argmax(action.detach().numpy()[0])
    next_state,reward,done,_=env.step(action)
    if i%50==0:
        print(i,"   ",reward,done)
    state=next_state

save_frames_as_gif(frames, filename="plots/CartPole.gif")
    
env.close()
