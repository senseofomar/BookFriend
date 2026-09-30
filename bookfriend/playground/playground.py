from bookfriend.utils import session_utils

def main():
    path ="session.json"

    print("----Testing save session")
    session_utils.save_session({"user": "omar", "progress":42}, path)
    print("Saved session.json")

    print("\n---Testing load_session---")
    data = session_utils.load_session(path)
    print("Loaded data:",data)

    print("\n---testing Reset_session")
    session_utils.reset_session(path)
    data = session_utils.load_session(path)
    print("After reset, data:", data)

if __name__ == '__main__':
    main() 