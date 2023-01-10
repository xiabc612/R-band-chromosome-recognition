import os

def read_data(rootdir,rootdirval):
    rootdir=rootdir
    rootdirval=rootdirval

    valcasenames=os.listdir(rootdirval)
    valcasenames.sort()

    train1list1=os.listdir(os.path.join(rootdir,'1'))
    train2list1=os.listdir(os.path.join(rootdir,'2'))
    train3list1=os.listdir(os.path.join(rootdir,'3'))
    train4list1=os.listdir(os.path.join(rootdir,'4'))
    train5list1=os.listdir(os.path.join(rootdir,'5'))
    train6list1=os.listdir(os.path.join(rootdir,'6'))
    train7list1=os.listdir(os.path.join(rootdir,'7'))
    train8list1=os.listdir(os.path.join(rootdir,'8'))
    train9list1=os.listdir(os.path.join(rootdir,'9'))
    train10list1=os.listdir(os.path.join(rootdir,'10'))
    train11list1=os.listdir(os.path.join(rootdir,'11'))
    train12list1=os.listdir(os.path.join(rootdir,'12'))
    train13list1=os.listdir(os.path.join(rootdir,'13'))
    train14list1=os.listdir(os.path.join(rootdir,'14'))
    train15list1=os.listdir(os.path.join(rootdir,'15'))
    train16list1=os.listdir(os.path.join(rootdir,'16'))
    train17list1=os.listdir(os.path.join(rootdir,'17'))
    train18list1=os.listdir(os.path.join(rootdir,'18'))
    train19list1=os.listdir(os.path.join(rootdir,'19'))
    train20list1=os.listdir(os.path.join(rootdir,'20'))
    train21list1=os.listdir(os.path.join(rootdir,'21'))
    train22list1=os.listdir(os.path.join(rootdir,'22'))
    train23list1=os.listdir(os.path.join(rootdir,'23'))
    train24list1=os.listdir(os.path.join(rootdir,'24'))

    list1=[train1list1,train2list1,train3list1,train4list1,train5list1,train6list1,train7list1,train8list1,
    train9list1,train10list1,train11list1,train12list1,train13list1,train14list1,train15list1,train16list1,train17list1,
    train18list1,train19list1,train20list1,train21list1,train22list1,train23list1,train24list1]
    dirlist=['1/', '2/', '3/', '4/', '5/', '6/', '7/', '8/', '9/', '10/', '11/', '12/', '13/', '14/', '15/', '16/', '17/', '18/', '19/', '20/', '21/', '22/', '23/', '24/']
    normalboy=[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 23]
    normalgirl=[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22]
    reduced=0
    label2list=[0,0,0,1,1,2,2,2,2,2,2,2,3,3,3,4,4,4,5,5,6,6,2,6]

    return valcasenames,list1,dirlist,normalboy,normalgirl,reduced,label2list




