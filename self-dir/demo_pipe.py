import multiprocessing

def worker(conn):
    # 子进程从管道接收消息
    while True:
        msg = conn.recv()
        if msg == 'STOP':
            print(f'Worker received stop signal. Exiting.')
            break
        print(f'Worker received: {msg}')
        # 子进程向管道发送响应
        conn.send(f'Echo: {msg}')

if __name__ == '__main__':
    # 创建管道
    parent_conn, child_conn = multiprocessing.Pipe()

    # 创建子进程
    p = multiprocessing.Process(target=worker, args=(child_conn,))
    p.start()

    # 父进程向管道发送消息
    for i in range(5):
        msg = f'Message {i}'
        parent_conn.send(msg)
        # 父进程从管道接收响应
        response = parent_conn.recv()
        print(f'Parent received: {response}')

    # 发送停止信号
    parent_conn.send('STOP')
    p.join()

    print('Parent process finished.')