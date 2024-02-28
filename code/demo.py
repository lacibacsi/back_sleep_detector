
import cmd

# simple demo file for showing the model to David and Samuel

class BackTestDemo(cmd.Cmd):

    def do_help(self, line):
        print("use: train for training, predict <filename> for prediction on an image")

    def do_train(self, line):
        print("starting training on default dataset")


    def do_EOF(self, line):
        return True


if __name__ == '__main__':
    BackTestDemo().cmdloop()