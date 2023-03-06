"""
Name : test_api.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2023/3/6 12:15
Desc:
"""
import requests
import threading
import time

if __name__ == '__main__':
    # 并发请求数量
    concurrent_num = 1

    input_datas = [{'title': '',
                    'document': '关于丽湖校区北区楼宇11月12日凌晨时段停水的通知各位师生：因四方楼水泵房水箱进水改管需要，丽湖校区部分楼宇供水将受到影响，具体影响时间及影响范围如下：影响时间：11月12日（星期六）00：00-11月12日（星期六）7:00影响范围：梧桐树、青冈栎、三角梅、冬青树、紫罗兰、伐木餐厅、伐檀餐厅、留学生活动中心、公共教学楼(四方楼）、明理楼、明德楼、明律楼、启明楼（中央图书馆）、守正楼。请各涉及楼栋师生注意做好停水准备，给您带来不便，敬请谅解！如有疑问，请联系中航物业（丽湖校区）24小时客服中心值班电话：0755-21672017。丽湖校区管理办公室2022年11月10日关于丽湖校区北区楼宇11月12日凌晨时段停水的通知各位师生：因四方楼水泵房水箱进水改管需要，丽湖校区部分楼宇供水将受到影响，具体影响时间及影响范围如下：影响时间：11月12日（星期六）00：00-11月12日（星期六）7:00影响范围：梧桐树、青冈栎、三角梅、冬青树、紫罗兰、伐木餐厅、伐檀餐厅、留学生活动中心、公共教学楼(四方楼）、明理楼、明德楼、明律楼、启明楼（中央图书馆）、守正楼。请各涉及楼栋师生注意做好停水准备，给您带来不便，敬请谅解！如有疑问，请联系中航物业（丽湖校区）24小时客服中心值班电话：0755-21672017。丽湖校区管理办公室2022年11月10日',
                    'document_id': 'life_3_seg_0',
                    'question': '宿舍为什么停水？'}
                   ] * 10

    # 测试函数
    def test_qa_api(idx):
        url = 'http://172.17.0.7:8080/qa'
        timestamp = time.time()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        print(f'线程{idx}开始时间：{formatted_time}')
        response = requests.post(url, json=input_datas)
        end_timestamp = time.time()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_timestamp))
        print(f'线程{idx}结束时间：{formatted_time}')
        print(f'线程{idx}花费时间：{end_timestamp - timestamp}')
        if response.ok:
            print(response.json())
        else:
            print(response.text)


    # 多线程测试
    threads = []
    for i in range(concurrent_num):
        t = threading.Thread(target=test_qa_api, kwargs={'idx': i})
        threads.append(t)
        t.start()

    # 等待所有线程执行完毕
    for t in threads:
        t.join()
