import pygame, sys ,os,neat,pickle
from pygame.locals import *

#pygame window size
WIDTH=800    
HEIGHT=800

#generation number
gen=0

#Level class to draw the level
class Level():
    def __init__(self):
        self.blocks=[100,300,500,700]
        self.window=[(400,500),(200,400),(400,600),(300,500)]
        self.rects=[pygame.rect.Rect(100,0,20,400),pygame.rect.Rect(100,500,20,300),
            pygame.rect.Rect(300,0,20,200),pygame.rect.Rect(300,400,20,400), pygame.rect.Rect(500,0,20,400), pygame.rect.Rect(500,600,20,200),
             pygame.rect.Rect(700,0,20,300), pygame.rect.Rect(700,500,20,300)]
    
    def draw(self,display):
        for r in self.rects:
            pygame.draw.rect(display,(0,0,0),r)

#Player class that is controlled by the AI
class Blob():
    def __init__(self):
        self.x=30
        self.y=350
        self.passed=0      
        self.last_passed=0
        self.prev_dir=1
    
    def get_rect(self):
        return pygame.rect.Rect(self.x,self.y,20,20)

    def draw(self,display):
        pygame.draw.rect(display,(0,0,255),(self.x,self.y,20,20))
    
    def move(self,dir):
        if dir==1:
            self.x+=5
        if dir==2:
            self.y-=5
        if dir==3:
            self.x-=5
        if dir==4:
            self.y+=5

#init pygame
pygame.init()
display=pygame.display.set_mode((WIDTH,HEIGHT))

level=Level()
blob=Blob()

#function to evaluate the genomes
def eval_genomes(genomes,config):

    global gen
    gen+=1
    game_over=False
    blobs=[]
    nets=[]
    ge=[]

    #init genomes,nets and blobs
    for genome_id,genome in genomes:
        genome.fitness=0
        net=neat.nn.FeedForwardNetwork.create(genome,config)
        blobs.append(Blob())
        nets.append(net)
        ge.append(genome)
    
    #run loop until all blobs die or game_over =True
    while len(blobs)>0 and game_over==False:
        for e in pygame.event.get():
            if e.type==QUIT:
                pygame.quit()
                exit()


        for x,blob in enumerate(blobs):
            ge[x].fitness-=0.1    #decrease fitness of each blob for each frame it stays alive to avoid loops at one place

            #calcutate no of obstacles passed
            passed=0
            for b in level.blocks:
                if blob.x>b:
                    passed+=1
            blob.passed=passed

            #give fitness for passing obstacles
            if blob.passed>0 and blob.passed-blob.last_passed>0:
                ge[x].fitness+=(blob.passed)*50
                blob.last_passed=blob.passed
            
            #calculate next open window for input to the network
            if passed<4:
                window=level.window[passed]
                window_top=window[0]
                window_bottom=window[1]
            else:
                game_over=True #id all obstacles passed then game_over=True

            #get output from the neural network which decides where to move the blob
            output=nets[blobs.index(blob)].activate(((blob.y-window_bottom),(blob.y-window_top),abs(blob.x-level.blocks[passed]) if passed <4 else 0))

            #decide the output and move the blob accordingly
            m=max(output)

            if output[0]==m:
                if blob.prev_dir!=3:
                    blob.move(1)
                    blob.prev_dir=1
                else :
                    blob.move(3)
                    blob.prev_dir=3
                
            elif output[1]==m:
                if blob.prev_dir!=4:
                    blob.move(2)
                    blob.prev_dir=2
                else:
                    blob.move(4)
                    blob.prev_dir=4               
                
            elif output[2]==m:
                if blob.prev_dir!=1:
                    blob.move(3)
                    blob.prev_dir=3 
                else :
                    blob.move(1)
                    blob.prev_dir=1 
            elif output[3]==m:
                if blob.prev_dir!=2:
                    blob.move(4)
                    blob.prev_dir=4
                else :
                    blob.move(2)
                    blob.prev_dir=2

            #remove the blob if it is looping around 
            if ge[x].fitness<-100:
                nets.pop(blobs.index(blob))
                ge.pop(blobs.index(blob))
                blobs.pop(blobs.index(blob))       

        #detect collisions
        for r in level.rects:
            for blob in blobs:
                if r.colliderect(pygame.rect.Rect(blob.x,blob.y,20,20)):
                    #remove blob and decrease fitness
                    ge[blobs.index(blob)].fitness-=50
                    nets.pop(blobs.index(blob))
                    ge.pop(blobs.index(blob))
                    blobs.pop(blobs.index(blob))

                if blob.x<10 or blob.y<10 or blob.y>770:
                    #remove blob and decrease fitness
                    ge[blobs.index(blob)].fitness-=50
                    nets.pop(blobs.index(blob))
                    ge.pop(blobs.index(blob))
                    blobs.pop(blobs.index(blob))

        #update the display 
        display.fill([255,255,255])
        level.draw(display)
        for blob in blobs:      
            blob.draw(display)
        pygame.display.flip()


def run(config_file):
    """
    runs the NEAT algorithm to train a neural network.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 200 generations.
    winner = p.run(eval_genomes,200)
    
    #store the best genome
    with open('winner.pickle','wb') as f:
        pickle.dump(winner,f)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)
