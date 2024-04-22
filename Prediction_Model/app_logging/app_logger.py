from datetime import datetime

class App_Logger:
    def __init__(self) -> None:
        pass

    def add_log(self, file_object, message):
        self.current_date = datetime.now().date()
        self.current_time = datetime.now().time().strftime("%H:%M:%S")
        file_object.write(str(self.current_date) + "\t" + str(self.current_time) + "\t\t" + message + "\n")