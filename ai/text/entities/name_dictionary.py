COMMON_NAMES = {
    "monika",
    "harshith",
    "harshitha",
    "nivrithi",
    "pranav",
    "arjun",
    "kavya",
    "sanjay",
    "suresh",
    "ramesh",
    "divya",
    "priya",
    "karthik",
    "vijay",
    "ajith",
    "sneha",
    "rohit",
    "rahul",
    "pooja",
    "neha",
    "swathi",
    "vikram",
    "aravind",
    "siddharth",
    "meera",
    "ananya",
    "krish",
    "deepak",
    "varun",
    "akash",
    "surya",
    "prakash",
    "naveen",
    "arathi",
    "bharath",
    "gokul",
    "hari",
    "jayasri",
    "kamal",
    "lavanya",
    "mithun",
    "nandini",
    "omkar",
    "preetha",
    "qureshi",
    "rajesh",
    "sriram",
    "tarun",
    "uma",
    "vishal",
    "yash",
    "zoya",

    # --- Additional common Indian names, grouped by region for maintainability.
    # Curated, well-known first names -- not exhaustive, and not a claim of
    # correctness for any individual name; used only as a fuzzy-matching
    # reference set by ai/voice/name_protection.py, never as ground truth.

    # Tamil
    "arun", "karthikeyan", "murugan", "meena", "lakshmi", "selvam",
    "kannan", "muthu", "shanmugam", "kumaran", "ilango", "senthil",
    "elango", "thangam", "valli", "ponni", "veera", "manikandan",
    "sundaram", "krishnamoorthy", "chandran", "ganesan", "balasubramaniam",
    "kavitha", "geetha", "revathi", "anitha", "mahalakshmi", "saravanan",

    # Malayalam
    "biju", "anoop", "remya", "sreelakshmi", "manoj", "sudheer",
    "anil", "sajitha", "reshma", "arun" , "vinod", "rajan",
    "sreekumar", "beena", "shyamala", "gopan", "nandakumar", "sreeja",
    "jayan", "anu", "sabu", "prasad", "sindhu", "shibu",

    # Telugu
    "venkatesh", "srinivas", "padma", "lakshmi", "ramana", "satyanarayana",
    "nagesh", "vijayalakshmi", "chandrasekhar", "prasanna", "naveen",
    "srilatha", "raghava", "sailaja", "sarita", "koteswara", "anitha",
    "venkataramana", "sujatha", "ramarao", "subbarao",

    # Kannada
    "manju", "shivakumar", "chaitra", "raghavendra", "nataraj",
    "veeresh", "basavaraj", "puneeth", "shruthi", "sowmya", "manjunath",
    "girish", "kavya", "nataraja", "shivanna", "suma", "rekha",

    # Hindi / North Indian
    "amit", "sunita", "rajesh", "pooja", "anil", "rekha", "suresh",
    "mahesh", "rakesh", "ashok", "vinod", "manoj", "sanjeev", "geeta",
    "kavita", "seema", "rina", "manisha", "vikas", "sandeep", "deepika",
    "ravi", "sunil", "pankaj", "anjali", "neetu", "arvind", "yogesh",

    # Common standalone Indian surnames -- kept in the same flat set (not a
    # separate list) so a bare surname like "Kumar" is recognized as
    # already-correct instead of being fuzzy-matched to an unrelated first
    # name that happens to share a prefix (e.g. "Kumaran").
    "kumar", "kumari", "devi", "reddy", "rao", "nair", "iyer", "pillai",
    "menon", "gowda", "naidu", "chetty", "singh", "sharma", "gupta",
    "verma", "patel", "shah", "yadav", "chauhan", "iyengar", "pillai",
    "nambiar", "warrier", "achari", "chettiar", "mudaliar", "naicker",
}
