
# 主流程函数
def analyze_project(self,project_folder: str, output_folder: str)->str:
    """
    Analyze all Python files in the specified directory and provide a summary.

    Args:
        self (Agent): The agent instance calling the function.
        project_folder (str): The path to the directory containing Python files to be analyzed.
        output_folder (str):The path of output file.

    Returns:
        str: A summary analysis result for the entire project. 
    """
    import os
    import json
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from typing import List, Dict  
    # 单文件分析任务
    MERMAID_TEMPLATE = r"""
```mermaid
flowchart LR
    %% <gpt_academic_hide_mermaid_code> 一个特殊标记，用于在生成mermaid图表时隐藏代码块
    classDef Comment stroke-dasharray: 5 5
    subgraph {graph_name}
{relationship}
    end
```
"""
    def analyze_single_file(file_path: str) -> Dict:
        import os
        import json
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from typing import List, Dict  
        """使用 LLM 分析单个文件"""
        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.read()

        # 构造消息格式
        messages = [
    {"role": "system", "content": "You are a software architecture analyst analyzing a source code project. Your responses must be clear and concise."},
    {"role": "user", "content": f"Please provide an overview of the following program file. The file name is {file_path}, and the file content is {file_content}"}
]


        # 调用 API 并返回结果
        response = llm_client.generate_response(messages)
        return {
            "file": file_path,
            "analysis": response["content"]
        }

    def indent(text, prefix, predicate=None):
        import os
        import json
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from typing import List, Dict  
        """Adds 'prefix' to the beginning of selected lines in 'text'.

        If 'predicate' is provided, 'prefix' will only be added to the lines
        where 'predicate(line)' is True. If 'predicate' is not provided,
        it will default to adding 'prefix' to all non-empty lines that do not
        consist solely of whitespace characters.
        """
        if predicate is None:
            def predicate(line):
                return line.strip()

        def prefixed_lines():
            for line in text.splitlines(True):
                yield (prefix + line if predicate(line) else line)
        return ''.join(prefixed_lines())

    def build_file_tree_mermaid_diagram(file_manifest, file_comments, graph_name):
        import os
        import json
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from typing import List, Dict  
        class FileNode:
            def __init__(self, name):
                self.name = name
                self.children = []
                self.is_leaf = False
                self.level = 0
                self.parenting_ship = []
                self.comment = ""
                self.comment_maxlen_show = 50

            @staticmethod
            def add_linebreaks_at_spaces(string, interval=10):
                return '\n'.join(string[i:i+interval] for i in range(0, len(string), interval))

            def sanitize_comment(self, comment):
                if len(comment) > self.comment_maxlen_show: suf = '...'
                else: suf = ''
                comment = comment[:self.comment_maxlen_show]
                comment = comment.replace('\"', '').replace('`', '').replace('\n', '').replace('`', '').replace('$', '')
                comment = self.add_linebreaks_at_spaces(comment, 10)
                return '`' + comment + suf + '`'

            def add_file(self, file_path, file_comment):
                directory_names, file_name = os.path.split(file_path)
                current_node = self
                level = 1
                if directory_names == "":
                    new_node = FileNode(file_name)
                    current_node.children.append(new_node)
                    new_node.is_leaf = True
                    new_node.comment = self.sanitize_comment(file_comment)
                    new_node.level = level
                    current_node = new_node
                else:
                    dnamesplit = directory_names.split(os.sep)
                    for i, directory_name in enumerate(dnamesplit):
                        found_child = False
                        level += 1
                        for child in current_node.children:
                            if child.name == directory_name:
                                current_node = child
                                found_child = True
                                break
                        if not found_child:
                            new_node = FileNode(directory_name)
                            current_node.children.append(new_node)
                            new_node.level = level - 1
                            current_node = new_node
                    term = FileNode(file_name)
                    term.level = level
                    term.comment = self.sanitize_comment(file_comment)
                    term.is_leaf = True
                    current_node.children.append(term)

            def print_files_recursively(self, level=0, code="R0"):
                print('    '*level + self.name + ' ' + str(self.is_leaf) + ' ' + str(self.level))
                for j, child in enumerate(self.children):
                    child.print_files_recursively(level=level+1, code=code+str(j))
                    self.parenting_ship.extend(child.parenting_ship)
                    p1 = f"""{code}[\"🗎{self.name}\"]""" if self.is_leaf else f"""{code}[[\"📁{self.name}\"]]"""
                    p2 = """ --> """
                    p3 = f"""{code+str(j)}[\"🗎{child.name}\"]""" if child.is_leaf else f"""{code+str(j)}[[\"📁{child.name}\"]]"""
                    edge_code = p1 + p2 + p3
                    if edge_code in self.parenting_ship:
                        continue
                    self.parenting_ship.append(edge_code)
                if self.comment != "":
                    pc1 = f"""{code}[\"🗎{self.name}\"]""" if self.is_leaf else f"""{code}[[\"📁{self.name}\"]]"""
                    pc2 = f""" -.-x """
                    pc3 = f"""C{code}[\"{self.comment}\"]:::Comment"""
                    edge_code = pc1 + pc2 + pc3
                    self.parenting_ship.append(edge_code)

        # Create the root node
        file_tree_struct = FileNode("root")
        # Build the tree structure
        for file_path, file_comment in zip(file_manifest, file_comments):
            file_tree_struct.add_file(file_path, file_comment)
        file_tree_struct.print_files_recursively()
        cc = "\n".join(file_tree_struct.parenting_ship)
        ccc = indent(cc, prefix=" "*8)
        return MERMAID_TEMPLATE.format(graph_name=graph_name, relationship=ccc)
    # 输入处理
    def get_file_manifest(project_folder: str) -> List[str]:
        import os
        import json
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from typing import List, Dict    
        """获取目录下的 Python 文件列表"""
        if not os.path.exists(project_folder):
            raise FileNotFoundError(f"filepath {project_folder} not exist!")

        file_manifest = []
        for root, _, files in os.walk(project_folder):
            for file in files:
                if file.endswith(".py"):
                    file_manifest.append(os.path.join(root, file))
        if not file_manifest:
            raise FileNotFoundError("no Python file!")

        if len(file_manifest) > 512:
            raise ValueError("over 512!")
        return file_manifest
# 多线程逐文件分析
    def analyze_files_multithread(file_manifest: List[str]) -> List[Dict]:
        import os
        import json
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from typing import List, Dict  
        """多线程分析文件"""
        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(analyze_single_file, file): file for file in file_manifest}
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"error when analyzing {futures[future]} : {e}")
        return results
    
    class LLMClient:
        def __init__(self, base_url: str, api_key: str, model: str):
            from openai import OpenAI
            self.client = OpenAI(base_url=base_url, api_key=api_key)
            self.model = model

        def generate_response(self, messages: List[Dict[str, str]]) -> Dict:
            """
            调用 Sambanova LLM API 生成响应。
            :param messages: 消息列表，包含 role 和 content 键。
            :return: LLM 的生成结果，包含生成的内容和使用情况。
            """
            try:
                # 调用 API 生成响应
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )

                # 检查响应是否包含生成的内容
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    message_content = response.choices[0].message.content
                    # usage_info = response.usage if hasattr(response, 'usage') else {}
                    return {
                        "content": message_content
                    }
                else:
                    raise ValueError("No content generated in the response!")

            except Exception as e:
                print(f"Error occurred while calling the API: {e}")
                return {"content": "Call failed, please check the input or service status.", "usage": {}}

    llm_client = LLMClient(
            base_url="https://api.sambanova.ai/v1",
            api_key="614e1948-9d06-4764-8124-9cad201c8281",
            model="Meta-Llama-3.1-8B-Instruct"
        )
    # Step 1: 获取文件清单
    file_manifest = get_file_manifest(project_folder)
    print(f"发现 {len(file_manifest)} 个文件，开始分析...")

    # Step 2: 多线程逐文件分析
    analysis_results = analyze_files_multithread(file_manifest)
    print("逐文件分析完成，保存中间结果...")
    with open(os.path.join(output_folder, "file_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=4)
    # 批量汇总分析
    def summarize_files_in_batches(analysis_results: List[Dict], batch_size: int = 16) -> List[Dict]:
        import os
        import json
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from typing import List, Dict  
        """按批次汇总文件分析结果"""
        summaries = []
        for i in range(0, len(analysis_results), batch_size):
            batch = analysis_results[i:i + batch_size]
            file_descriptions = "\n".join([f"{res['file']}: {res['analysis']}" for res in batch])

            # 构造消息格式
            messages = [
    {"role": "system", "content": "You are a software architecture analyst, analyzing the source code of a project."},
    {"role": "user", "content": f"Briefly describe the functionality of the following files using a Markdown table:\n{file_descriptions}\nBased on the above analysis, summarize the overall functionality of the program in one sentence."}
]


            # 调用 API 并保存结果
            response = llm_client.generate_response(messages)
            summaries.append({
                "batch_start": i,
                "batch_end": i + batch_size,
                "summary": response["content"]
            })
        return summaries
    # 可视化文件树
    def generate_file_tree_diagram(project_folder: str, file_manifest: List[str], file_comments: List[str]) -> str:
        import os
        import json
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from typing import List, Dict  
        """
        生成项目文件树的 Mermaid.js 图表。
        :param project_folder: 项目根目录。
        :param file_manifest: 文件路径列表。
        :param file_comments: 对应文件的评论或分析。
        :return: Mermaid.js 格式的文件树图表。
        """
        graph_name = "项目文件树"
        diagram_code = build_file_tree_mermaid_diagram(file_manifest, file_comments, graph_name)
        return diagram_code

    # Step 3: 批量汇总分析
    summaries = summarize_files_in_batches(analysis_results)
    print("汇总分析完成，保存结果...")
    with open(os.path.join(output_folder, "summary_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=4)

    # 提取汇总内容为 Markdown 表格
    markdown_summary = "| File | Analysis |\n| --- | --- |\n"
    for result in analysis_results:
        markdown_summary += f"| {os.path.basename(result['file'])} | {result['analysis']} |\n"

    # Step 4: 生成文件树可视化
    # 从分析结果中提取文件注释（分析内容）
    file_comments = [result["analysis"] for result in analysis_results]
    file_tree_diagram = generate_file_tree_diagram(project_folder, file_manifest, file_comments)
    with open(os.path.join(output_folder, "file_tree.md"), "w", encoding="utf-8") as f:
        f.write(file_tree_diagram)

    return (
            f"Analysis complete! The result has been saved to {output_folder}\n\n"
            f"### File Tree:\n{file_tree_diagram}\n\n"
            f"### File Analysis Summary:\n{markdown_summary}"
            f"请以markdown格式展现File Analysis Summary"
            )

