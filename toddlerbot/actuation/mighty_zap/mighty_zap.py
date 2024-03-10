import serial

PROTOCOL_TX_BUF_SIZE = 50
PROTOCOL_RX_BUF_SIZE = 50
MIGHTYZAP_PING = 0xF1
MIGHTYZAP_READ_DATA = 0xF2
MIGHTYZAP_WRITE_DATA = 0xF3
MIGHTYZAP_REG_WRITE = 0xF4
MIGHTYZAP_ACTION = 0xF5
MIGHTYZAP_RESET = 0xF6
MIGHTYZAP_RESTART = 0xF8
MIGHTYZAP_FACTORY_RESET = 0xF9
MIGHTYZAP_SYNC_WRITE = 0x73

TxBuffer = [0] * PROTOCOL_TX_BUF_SIZE
TxBuffer_index = 0
RxBuffer = [0] * PROTOCOL_RX_BUF_SIZE
RxBuffer_size = 0

ErollService = 0
ErollService_Instruction = 0
ErollService_ID = 0x00
ErollService_Addr = 0x00
ErollService_Size = 0x00
ErollService_ModelNum = 0x0000
ActuatorID = 0
checksum = 0
MZap = serial.Serial()


def SetProtocalHeader():
    global TxBuffer_index
    global TxBuffer
    TxBuffer_index = 0
    TxBuffer[TxBuffer_index] = 0xFF
    TxBuffer_index += 1
    TxBuffer[TxBuffer_index] = 0xFF
    TxBuffer_index += 1
    TxBuffer[TxBuffer_index] = 0xFF
    TxBuffer_index += 1
    TxBuffer[TxBuffer_index] = ActuatorID
    TxBuffer_index += 1


def SetProtocolInstruction(ins):
    global TxBuffer_index
    global TxBuffer
    global ErollService_Instruction

    TxBuffer_index = 5
    ErollService_Instruction = ins
    TxBuffer[TxBuffer_index] = ins
    TxBuffer_index += 1


def AddProtocolFactor(para):
    global TxBuffer_index
    global TxBuffer
    TxBuffer[TxBuffer_index] = para
    TxBuffer_index += 1


def SetProtocollength_checksum():
    global TxBuffer_index
    global TxBuffer
    global checksum
    checksum = 0
    start_i = 0

    TxBuffer[4] = TxBuffer_index - 4
    start_i = 3

    for i in range(start_i, TxBuffer_index):
        checksum += TxBuffer[i]
    TxBuffer[TxBuffer_index] = (checksum & 0x000000FF) ^ 0x000000FF
    TxBuffer_index += 1


def getID():
    global ActuatorID
    return ActuatorID


def setID(ID):
    global ActuatorID
    ActuatorID = ID


def MightyZap(ID):
    global ErollService
    global ErollService_Instruction
    global ErollService_ID
    global ErollService_Addr
    global ErollService_Size
    global ErollService_Size

    ErollService = 0
    ErollService_Instruction = 0
    ErollService_ID = 0x00
    ErollService_Addr = 0x00
    ErollService_Size = 0x00
    ErollService_ModelNum = 0x0000

    setID(ID)


def OpenMightyZap(portname, BaudRate):
    MZap.port = portname
    MZap.baudrate = BaudRate
    MZap.timeout = 0.1

    MZap.open()


def serialtimeout(time):
    MZap.timeout = time


def CloseMightyZap():
    MZap.close()


def SendPacket():
    global TxBuffer_index
    global TxBuffer
    for i in range(0, TxBuffer_index):
        MZap.write([TxBuffer[i]])


def ReceivePacket(ID, size):
    global TxBuffer_index
    global TxBuffer
    global RxBuffer
    timeout = 0
    temp = 0
    i = 0
    head_count = 0

    while head_count < 3:
        timeout = timeout + 1
        if timeout > 5:
            RxBuffer[6] = 0
            RxBuffer[7] = 0
            return -1

        temp = MZap.read(1)

        if temp == b"\xff":
            RxBuffer[head_count] = 0xFF
            head_count += 1
        else:
            RxBuffer[0] = 0
            head_count = 0

    for i in range(3, size):
        temp = ord(MZap.read(1))
        RxBuffer[i] = temp

    return 1


def read_data(ID, addr, size):
    global MIGHTYZAP_READ_DATA

    setID(ID)
    SetProtocalHeader()
    SetProtocolInstruction(MIGHTYZAP_READ_DATA)
    AddProtocolFactor(addr)
    AddProtocolFactor(size)
    SetProtocollength_checksum()
    SendPacket()


def ead_data_model_num(ID):
    global MIGHTYZAP_READ_DATA
    setID(ID)
    SetProtocalHeader()
    SetProtocolInstruction(MIGHTYZAP_READ_DATA)
    AddProtocolFactor(0)
    ErollService_Addr = 0
    AddProtocolFactor(2)
    ErollService_Size = 2
    SetProtocollength_checksum()
    SendPacket()


def write_data(ID, addr, data, size):
    global MIGHTYZAP_WRITE_DATA
    i = 0
    setID(ID)
    SetProtocalHeader()
    SetProtocolInstruction(MIGHTYZAP_WRITE_DATA)
    AddProtocolFactor(addr)
    for i in range(0, size):
        AddProtocolFactor(data[i])
    SetProtocollength_checksum()
    SendPacket()


def Sync_write_data(addr, data, size):
    global MIGHTYZAP_SYNC_WRITE
    i = 0
    setID(0xFE)
    SetProtocalHeader()
    SetProtocolInstruction(MIGHTYZAP_SYNC_WRITE)
    AddProtocolFactor(addr)
    for i in range(0, size):
        AddProtocolFactor(data[i])
    SetProtocollength_checksum()
    SendPacket()


def WritePacket(buff, size):
    for i in range(0, size):
        MZap.write([buff[i]])


def action(ID):
    global MIGHTYZAP_WRITE_DATA
    setID(ID)
    SetProtocalHeader()
    SetProtocolInstruction(MIGHTYZAP_ACTION)
    SetProtocollength_checksum()
    SendPacket()


def reset_write(ID, option):
    global MIGHTYZAP_RESET
    setID(ID)
    SetProtocalHeader()
    SetProtocolInstruction(MIGHTYZAP_RESET)
    AddProtocolFactor(option)
    SetProtocollength_checksum()
    SendPacket()


def Restart(ID):
    global MIGHTYZAP_RESTART
    setID(ID)
    SetProtocalHeader()
    SetProtocolInstruction(MIGHTYZAP_RESTART)
    SetProtocollength_checksum()
    SendPacket()


def factory_reset_write(ID, option):
    global MIGHTYZAP_FACTORY_RESET
    setID(ID)
    SetProtocalHeader()
    SetProtocolInstruction(MIGHTYZAP_FACTORY_RESET)
    AddProtocolFactor(option)
    SetProtocollength_checksum()
    SendPacket()


def ping(ID):
    global MIGHTYZAP_PING
    setID(ID)
    SetProtocalHeader()
    SetProtocolInstruction(MIGHTYZAP_PING)
    SetProtocollength_checksum()
    SendPacket()


def changeID(bID, data):
    pByte = [0] * 1
    pByte[0] = data & 0x00FF
    setID(pByte[0])
    write_data(bID, 0x03, pByte, 1)


def GoalPosition(bID, position):
    pByte = [0] * 2
    pByte[0] = position & 0x00FF
    pByte[1] = position >> 8
    write_data(bID, 0x86, pByte, 2)


def PresentPosition(bID):

    global RxBuffer
    read_data(bID, 0x8C, 2)
    timeout = ReceivePacket(bID, 9)
    if timeout == 1:
        return (RxBuffer[7] * 256) + (RxBuffer[6])
    else:
        return -1


def GoalSpeed(bID, speed):
    pByte = [0] * 2

    pByte[0] = speed & 0x00FF
    pByte[1] = speed >> 8
    write_data(bID, 0x15, pByte, 2)


def GoalCurrent(bID, curr):
    pByte = [0] * 2

    pByte[0] = curr & 0x00FF
    pByte[1] = curr >> 8
    write_data(bID, 0x34, pByte, 2)


def Acceleration(bID, acc):
    pByte = [0] * 1
    pByte[0] = acc
    write_data(bID, 0x21, pByte, 1)


def Deceleration(bID, acc):
    pByte = [0] * 1
    pByte[0] = acc
    write_data(bID, 0x22, pByte, 1)


def ShortStrokeLimit(bID, SStroke):
    pByte = [0] * 2

    pByte[0] = SStroke & 0x00FF
    pByte[1] = SStroke >> 8
    write_data(bID, 0x06, pByte, 2)


def LongStrokeLimit(bID, LStroke):
    pByte = [0] * 2

    pByte[0] = LStroke & 0x00FF
    pByte[1] = LStroke >> 8
    write_data(bID, 0x08, pByte, 2)


def ForceEnable(bID, enable):
    pByte = [0] * 2

    if enable == 1:
        pByte[0] = 1
    else:
        pByte[0] = 0

    write_data(bID, 0x80, pByte, 1)
    SendPacket()


def SetShutDownEnable(bID, flag):
    pByte = [0] * 1
    pByte[0] = flag
    write_data(bID, 0x12, pByte, 1)


def GetShutDownEnable(bID):
    read_data(bID, 0x12, 1)
    timeout = ReceivePacket(bID, 8)
    if timeout == 1:
        return RxBuffer[6]
    else:
        return -1


def SetErrorIndicatorEnable(bID, flag):
    pByte = [0] * 1
    pByte[0] = flag
    write_data(bID, 0x11, pByte, 1)


def GetErrorIndicatorEnable(bID):
    read_data(bID, 0x11, 1)
    timeout = ReceivePacket(bID, 8)
    if timeout == 1:
        return RxBuffer[6]
    else:
        return -1


def ReadError(bID):
    ping(bID)
    timeout = ReceivePacket(bID, 7)
    if timeout == 1:
        return RxBuffer[5]
    else:
        return -1


def Write_Addr(bID, addr, size, data):
    if size == 2:
        pByte = [0] * 2
        pByte[0] = data & 0x00FF
        pByte[1] = data // 256
        write_data(bID, addr, pByte, 2)
    else:
        pByte = [0] * 1
        pByte[0] = data
        write_data(bID, addr, pByte, 1)


def Read_Addr(bID, addr, size):
    if size == 2:
        read_data(bID, addr, 2)
        timeout = ReceivePacket(bID, 9)
        if timeout == 1:
            return (RxBuffer[7] * 256) + RxBuffer[6]
        else:
            return -1
    else:
        read_data(bID, addr, 1)
        timeout = ReceivePacket(bID, 8)
        if timeout == 1:
            return RxBuffer[6]
        else:
            return -1
