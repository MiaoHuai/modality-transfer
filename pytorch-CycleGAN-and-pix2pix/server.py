import time
import threading
import uuid
import datetime
from queue import Queue
from flask import Flask, jsonify, request
from run import Runner

app = Flask(__name__)

# 全局任务队列和任务状态字典
task_queue = Queue()
task_status = {}

# 定期清理超过12小时的任务状态
def cleanup_task_status():
    print("Cleanup thread started.")
    while True:
        for task_id, task_info in task_status.items():
            if (datetime.datetime.now() - task_info['start_time']).total_seconds() > 12 * 3600:
                del task_status[task_id]
        time.sleep(3600)


# 模拟任务的执行过程
def execute_task(task_info, runner: Runner):
    task_id = task_info['task_id']
    task_data = task_info['task_data']
    results = []
    try:
        # 将任务状态改为进行中
        task_status[task_id]['status'] = 'Running'
        print(f"Task {task_id} is running...")

        # 执行任务
        for config in task_data['configs']:
            out_img_path = runner.run_batch(config)
            # 在循环体内部，对于 task_data 中的每个 config（即每个任务配置），调用 runner 实例的 run_batch 方法
            results.append({
                'configId': config['configId'],
                'filePaths': config['filePaths'],
                'file2FileList': out_img_path, 
                'modalityAugmentConfig':config["modalityAugmentConfig"]
            })
              
        # 执行完成后更新状态
        task_status[task_id]['status'] = 'Finshed'
        task_status[task_id]['results'] = results
        print(f"Task {task_id} is completed.")
        
    except Exception as e:
        # 如果执行过程中出错，记录错误并更新任务状态
        task_status[task_id]['status'] = 'error'
        task_status[task_id]['errorMsg'] = str(e)
        print(f"Task {task_id} failed with error: {e}")

# 任务执行器，轮询任务队列并执行任务
def task_worker():
    print("Task worker started.")
    runner = Runner()
    print("Runner initialized.")
    while True:
        # 从队列中取出任务（阻塞等待）
        task_info = task_queue.get()
        if task_info is None:
            break  # 终止条件
        execute_task(task_info, runner)
        task_queue.task_done()  # 标记任务完成

# Flask 路由，提交新任务
@app.route('/', methods=['POST'])
def submit_task():
    # 创建一个新的任务 ID
    task_id = str(uuid.uuid4())

    body = request.json
    try:
        for config in body['configs']:
            configId = config['configId']
            filePaths = config['filePaths']
            typeName = config['type']
            outputPath = config['outputPath']
            modalityAugmentConfig = config['modalityAugmentConfig']
            if configId is None or filePaths is None or typeName is None or outputPath is None or modalityAugmentConfig is None:
                raise ValueError('Invalid prameters')

    except Exception as e:
        return jsonify({
            'code' : 400,
            'data' : task_id,
            'error': str(e)
        }), 400

    
    # 将任务加入队列并初始化任务状态
    task_status[task_id] = {'status': 'Queue', 'start_time': datetime.datetime.now()}
    task_queue.put({
        'task_id': task_id,
        'task_data': body
    })
    
    # 返回任务 ID
    return jsonify({
        'code': 200,
        'data': task_id,
        'error': ''
    }), 200

# Flask 路由，查询任务状态
@app.route('/result', methods=['POST'])
def get_task_status():
    body = request.json
    task_id = body['taskId']
    if task_id in task_status:
        return jsonify({
            'code': 200,
            'data': task_status[task_id],
            'error': '' if 'errorMsg' not in task_status[task_id] is not None else task_status[task_id]['errorMsg']
        }), 200
    else:
        return jsonify({'error': 'Task ID not found'}), 404

# 启动任务执行线程
if __name__ == '__main__':
    # 启动后台任务处理线程
    worker_thread = threading.Thread(target=task_worker, daemon=False)
    worker_thread.start()

    # 启动清理任务状态线程
    cleanup_thread = threading.Thread(target=cleanup_task_status, daemon=False)
    cleanup_thread.start()
    
    # 启动 Flask 服务
    app.run(debug=True, host='0.0.0.0', port=5001)
