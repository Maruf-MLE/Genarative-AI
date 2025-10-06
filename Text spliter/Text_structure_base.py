from langchain.text_splitter import RecursiveCharacterTextSplitter

text ="""In 2025, cricket saw a blend of tradition and breakthrough: South Africa finally claimed their first World Test Championship (2023–25) title, defeating Australia in the final at Lord’s. 
Wikipedia
+1
 India reclaimed dominance in white-ball cricket by winning the ICC Champions Trophy, going unbeaten through the tournament and edging out New Zealand in the final. 
Wikipedia
+2
The Cricket Panda
+2
 In domestic franchise leagues, underdog stories emerged: Royal Challengers Bengaluru broke an 18-year drought to win the IPL, Fortune Barishal won the Bangladesh Premier League, and Mumbai Indians lifted the Women’s Premier League. 
Wikipedia
+3
OneCricket
+3
Wikipedia
+3
 Meanwhile, the Caribbean Premier League continued its entertainment streak with Trinbago Knight Riders capturing their fifth CPL title. 
Wikipedia
 Across many bilateral series, teams from Asia, especially India, Pakistan, Sri Lanka, and others, posted strong results, contributing to a resurgence of Asian cricket on the global stage.
 """

spliter= RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0

)

chunks= spliter.split_text(text)
print(len(chunks))

print(chunks[0])