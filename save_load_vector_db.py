import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# ১. ভেক্টর ডাটাবেস সেভ করার কোড:
# আপনার নোটবুকে যেখানে vectorstore (বা আপনার দেওয়া অন্য কোনো নাম) তৈরি হয়েছে, 
# তার ঠিক নিচে এই লাইনগুলো লিখে রান করবেন:

def save_my_vector_db(vectorstore, folder_name="my_faiss_index"):
    vectorstore.save_local(folder_name)
    print(f"ভেক্টর ডাটাবেস সফলভাবে '{folder_name}' ফোল্ডারে সেভ হয়েছে!")

# উদাহরণ: 
# save_my_vector_db(vectorstore)


# ২. পরবর্তীতে ডাটাবেস লোড করার কোড:
# নতুন করে আর পিডিএফ লোড বা চ্যাঙ্ক না করে সরাসরি সেভ করা ডাটাবেস লোড করুন:

def load_my_vector_db(folder_name="my_faiss_index"):
    # আপনার যেই এমবেডিং মডেল ব্যবহার করা হয়েছিল সেটি ডিফাইন করুন
    embeddings = OpenAIEmbeddings() 
    
    # allow_dangerous_deserialization=True দেওয়াটা আবশ্যক
    loaded_vectorstore = FAISS.load_local(
        folder_name, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    print(f"'{folder_name}' ফোল্ডার থেকে ভেক্টর ডাটাবেস সফলভাবে লোড হয়েছে!")
    return loaded_vectorstore

# উদাহরণ:
# new_vectorstore = load_my_vector_db()
