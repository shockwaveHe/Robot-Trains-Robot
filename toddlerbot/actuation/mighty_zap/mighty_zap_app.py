import glob
import sys

import serial
from PyQt5 import QtCore, QtWidgets, uic

from toddlerbot.actuation.mighty_zap import mighty_zap

QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
MightyZap = mighty_zap
isOpen = 0


class Form(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = uic.loadUi("toddlerbot/actuation/mighty_zap/mighty_zap_app.ui")
        self.ui.Baudrate.addItem("9600")
        self.ui.Baudrate.addItem("19200")
        self.ui.Baudrate.addItem("57600")
        self.ui.Baudrate.addItem("115200")
        self.ui.ServoID.setText("0")
        self.serial_ports()

        self.ui.Connect.clicked.connect(self.Connect)
        self.ui.GoalPosition.clicked.connect(self.GoalPosition)
        self.ui.SStrokieLimit.clicked.connect(self.ShortStrokeLimit)
        self.ui.LStrokeLimit.clicked.connect(self.LongStrokeLimit)
        self.ui.GoalSpeed.clicked.connect(self.GoalSpeed)
        self.ui.GoalCurrent.clicked.connect(self.GoalCurrent)
        self.ui.PresentPosition.clicked.connect(self.PresentPosition)
        self.ui.Write.clicked.connect(self.WriteMem)
        self.ui.Read.clicked.connect(self.ReadMem)
        self.ui.ErrorRead.clicked.connect(self.ErrorRead)
        self.ui.SDRead.clicked.connect(self.ShutDownRead)
        self.ui.SDWrite.clicked.connect(self.ShutDownWrite)

        self.ui.close.clicked.connect(self.exit)
        self.ui.show()

    def serial_ports(self):
        if sys.platform.startswith("win"):
            ports = ["COM%s" % (i + 1) for i in range(256)]
        elif sys.platform.startswith("linux") or sys.platform.startswith("cygwin"):
            # this excludes your current terminal "/dev/tty"
            ports = glob.glob("/dev/tty[A-Za-z]*")
        elif sys.platform.startswith("darwin"):
            ports = glob.glob("/dev/tty.*")
        else:
            raise EnvironmentError("Unsupported platform")

        result = []
        for port in ports:
            try:
                s = serial.Serial(port)
                s.close()
                self.ui.Port.addItem(port)
                result.append(port)
            except (OSError, serial.SerialException) as e:
                pass

        if len(result) == 0:
            print("1. Make sure you connect the linear actuator.")
            print("2. Try running 'sudo usermod -a -G dialout $USER' and then reboot.")

        return result

    def Connect(self):
        global isOpen
        if isOpen == 0:
            com = self.ui.Port.currentText()
            baud = self.ui.Baudrate.currentText()
            MightyZap.OpenMightyZap(com, baud)
            self.ui.Connect.setText("DisConnect")

            Servo_ID = int(self.ui.ServoID.toPlainText())
            position = MightyZap.PresentPosition(Servo_ID)

            Slimit = MightyZap.Read_Addr(Servo_ID, 0x06, 2)
            Llimit = MightyZap.Read_Addr(Servo_ID, 0x08, 2)
            G_Speed = MightyZap.Read_Addr(Servo_ID, 0x15, 2)
            G_Current = MightyZap.Read_Addr(Servo_ID, 0x34, 2)

            self.ui.txtPresentPosition.setText(str(position))
            self.ui.txtGoalPosition.setText(str(position))
            self.ui.txtSStrokeLimit.setText(str(Slimit))
            self.ui.txtLStrokeLimit.setText(str(Llimit))
            self.ui.txtGoalSpeed.setText(str(G_Speed))
            self.ui.txtGoalCurrent.setText(str(G_Current))

            isOpen = 1
        else:
            MightyZap.CloseMightyZap()
            self.ui.Connect.setText("Connect")
            isOpen = 0

    def GoalPosition(self):
        Servo_ID = int(self.ui.ServoID.toPlainText())
        position = int(self.ui.txtGoalPosition.toPlainText())
        MightyZap.GoalPosition(Servo_ID, position)

    def ShortStrokeLimit(self):
        Servo_ID = int(self.ui.ServoID.toPlainText())
        limit = int(self.ui.txtSStrokeLimit.toPlainText())
        MightyZap.Write_Addr(Servo_ID, 0x06, 2, limit)

    def LongStrokeLimit(self):
        Servo_ID = int(self.ui.ServoID.toPlainText())
        limit = int(self.ui.txtLStrokeLimit.toPlainText())
        MightyZap.Write_Addr(Servo_ID, 0x08, 2, limit)

    def GoalSpeed(self):
        Servo_ID = int(self.ui.ServoID.toPlainText())
        G_Speed = int(self.ui.txtGoalSpeed.toPlainText())
        MightyZap.Write_Addr(Servo_ID, 0x15, 2, G_Speed)

    def GoalCurrent(self):
        Servo_ID = int(self.ui.ServoID.toPlainText())
        G_Current = int(self.ui.txtGoalCurrent.toPlainText())
        MightyZap.Write_Addr(Servo_ID, 0x34, 2, G_Current)

    def PresentPosition(self):
        Servo_ID = int(self.ui.ServoID.toPlainText())
        position = MightyZap.PresentPosition(Servo_ID)
        self.ui.txtPresentPosition.setText(str(position))

    def WriteMem(self):
        Servo_ID = int(self.ui.ServoID.toPlainText())
        Addr = int(self.ui.writeAddr.toPlainText())
        AddrBytes = int(self.ui.writeAddrBytes.toPlainText())
        Data = int(self.ui.writeData.toPlainText())
        MightyZap.Write_Addr(Servo_ID, Addr, AddrBytes, Data)

    def ReadMem(self):
        Servo_ID = int(self.ui.ServoID.toPlainText())
        Addr = int(self.ui.readAddr.toPlainText())
        AddrBytes = int(self.ui.readAddrBytes.toPlainText())
        Data = MightyZap.Read_Addr(Servo_ID, Addr, AddrBytes)
        self.ui.ReadData.setText(str(Data))

    def ErrorRead(self):
        Servo_ID = int(self.ui.ServoID.toPlainText())
        Error = MightyZap.ReadError(Servo_ID)
        if (Error & 0x40) == 0x40:
            self.ui.InsError.setChecked(True)
        else:
            self.ui.InsError.setChecked(False)
        if (Error & 0x20) == 0x20:
            self.ui.OverloadError.setChecked(True)
        else:
            self.ui.OverloadError.setChecked(False)
        if (Error & 0x10) == 0x10:
            self.ui.ChecksumError.setChecked(True)
        else:
            self.ui.ChecksumError.setChecked(False)
        if (Error & 0x08) == 0x08:
            self.ui.RangeError.setChecked(True)
        else:
            self.ui.RangeError.setChecked(False)
        if (Error & 0x02) == 0x02:
            self.ui.StrokeError.setChecked(True)
        else:
            self.ui.StrokeError.setChecked(False)
        if (Error & 0x01) == 0x01:
            self.ui.VoltageError.setChecked(True)
        else:
            self.ui.VoltageError.setChecked(False)

    def ShutDownRead(self):
        Servo_ID = int(self.ui.ServoID.toPlainText())
        Error = MightyZap.GetShutDownEnable(Servo_ID)
        if (Error & 0x40) == 0x40:
            self.ui.InsSD.setChecked(True)
        else:
            self.ui.InsSD.setChecked(False)
        if (Error & 0x20) == 0x20:
            self.ui.OverloadSD.setChecked(True)
        else:
            self.ui.OverloadSD.setChecked(False)
        if (Error & 0x10) == 0x10:
            self.ui.CheckSumSD.setChecked(True)
        else:
            self.ui.CheckSumSD.setChecked(False)
        if (Error & 0x08) == 0x08:
            self.ui.RangeSD.setChecked(True)
        else:
            self.ui.RangeSD.setChecked(False)
        if (Error & 0x02) == 0x02:
            self.ui.StrokeSD.setChecked(True)
        else:
            self.ui.StrokeSD.setChecked(False)
        if (Error & 0x01) == 0x01:
            self.ui.VolatageSD.setChecked(True)
        else:
            self.ui.VolatageSD.setChecked(False)

    def ShutDownWrite(self):
        Servo_ID = int(self.ui.ServoID.toPlainText())

        Alarm = 0

        if self.ui.InsSD.isChecked() == True:
            Alarm |= 0x40
        if self.ui.OverloadSD.isChecked() == True:
            Alarm |= 0x20
        if self.ui.CheckSumSD.isChecked() == True:
            Alarm |= 0x10
        if self.ui.RangeSD.isChecked() == True:
            Alarm |= 0x08
        if self.ui.StrokeSD.isChecked() == True:
            Alarm |= 0x02
        if self.ui.VolatageSD.isChecked() == True:
            Alarm |= 0x01

        MightyZap.SetShutDownEnable(Servo_ID, Alarm)

    def exit(self):
        MightyZap.CloseMightyZap()
        sys.exit(app.exec())


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = Form()

    sys.exit(app.exec())
