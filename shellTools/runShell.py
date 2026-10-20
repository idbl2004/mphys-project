import subprocess
import uuid
import time

# class to open and run shell commands
class ShellRunner:
    """
    Class to represent a persistant shell for running HEASoft and Fermitools commands
    
    Attributes:
        __caldbinitString (str): command to initialize the CALDB environment
        __headasinitString (str): command to initialize the HEASoft environment
        errorLog (list[str]): list to hold error messages
        outputLog (list[str]): list to hold output messages

    Methods:
        runShellCommand(command: str) Runs a command in the shell process
        closeShell() Closes the shell process
    """
    # constructor to initialize the shell runner
    def __init__(self) -> None:
        self.outputLog: list[str] = []  # list to hold output messages
        self.__openShell()  # open the shell process

        # run the caldbinit and headasinit commands to initialize the environment
        caldbOutput: list[str] = self.runShellCommand(
            ". $CALDB/software/tools/caldbinit.sh",
            capture = False
            )  # run the caldbinit command
        headasOutput: list[str] = self.runShellCommand(
            ". $HEADAS/headas-init.sh",
            capture = False
            )  # run the headasinit command



    def __openShell(self) -> None:
        """Open a new shell process as an attribute of the class."""
        # open a new shell process as an atrubute of the class
        self.__shellProcess: subprocess.Popen = subprocess.Popen(
            ["bash"], # run a shell
            stdin = subprocess.PIPE, # pipe the input
            stdout = subprocess.PIPE, # pipe the output
            stderr = subprocess.STDOUT, # stdout the error
            text = True, # use text mode for input/output
            bufsize = 1 # line-buffered
        )


    def runShellCommand(
            self,
            command: str,
            capture: bool = True,
            ) -> list[str] | None:
        """Runs a command in the shell

        Args:
            command (str): The command to run in the shell
            capture (bool, optional): Whether to capture the output of the command. Defaults to True.

        Returns:
            list[str] | None: The output of the command as a list of strings, or None if capture is False.
        """

        # define a unique marker to indicate the end of the command output
        endMarker: str = f"__DONE__{uuid.uuid4()}__"

        # run the command in the shell process
        self.__shellProcess.stdin.write(command + "\n")
        self.__shellProcess.stdin.write(f"echo {endMarker}:$\n")
        self.__shellProcess.stdin.flush()

        # empty list to hold the output
        output: list[str] = []
        #returnCode: int = None

        # read the output of the command until the end marker is found
        while True:
            line: str = self.__shellProcess.stdout.readline()
            
            if not line:  # if the line is empty, the process has terminated
                time.sleep(0.01)
                continue

            line = line.strip()  # strip the line of whitespace
            if line.startswith(endMarker):  # if the line is the end marker, break the loop
                #returnCode = int(line.split(":")[1])  # get the return code from the end marker
                break

            self.outputLog.append(line)  # add the line to the output log

            if capture:
                output.append(line)  # add the line to the output list
            
        return output#, returnCode


    def closeShell(self) -> None:
        # close the shell process
        if self.__shellProcess and self.__shellProcess.poll() is None:  # check if the process is still running
            self.__shellProcess.stdin.write("exit\n")
            self.__shellProcess.stdin.flush()
            self.__shellProcess.wait()  # wait for the process to terminate






if __name__ == "__main__":
    # open a shell
    shell: ShellRunner = ShellRunner()

    #run a command to print the current working directory
    print("Current working directory:")
    output= shell.runShellCommand("pwd")
    print("Output:")
    for line in output:
        print(line)


    # run a command to list the files in the current directory
    print("\nFiles in the current directory:")
    output = shell.runShellCommand("ls -l")
    print("Output:")
    for line in output:
        print(line)


    # close the shell
    shell.closeShell()


