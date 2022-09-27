from kilt.knowledge_source import KnowledgeSource
import pandas as pd


if __name__ == "__main__":
    ks = KnowledgeSource()
    df = pd.read_json("data/totto/linearized/train.jsonl", lines=True)

    titles = df["table_page_title"].values.tolist()

    no_text_count = 0
    total_count = 0

    for title in titles:
        if "List" in title:
            page = ks.get_page_by_title(title)
            
            if page is not None:
                total_count += 1
                c = 0
                for passage in page["text"]:
                    if "::::" in passage or " to " in passage:
                        continue
                    elif len(passage) < 20:
                        continue
                    else:
                        c += 1
                    if c > 2:
                        break
                else:
                    no_text_count += 1
        if total_count % 20 == 0:
            print(total_count, no_text_count)

    print(no_text_count, total_count)