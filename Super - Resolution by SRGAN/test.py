import config
from torch import optim
from utils import load_checkpoint, plot_examples
from model import Generator
def test (gen):

    plot_examples("test_images/", gen)
    

def main(): 
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            opt_gen,
            config.LEARNING_RATE,
        )
    test(gen)
if __name__ == "__main__":
    main()