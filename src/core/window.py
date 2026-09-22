import pywinctl as pwc

class UniversalWindowObserver:
    def get_active_window_title(self) -> str:
        active_window = pwc.getActiveWindow()
        return active_window.title if active_window else "Desktop"

if __name__ == "__main__":
    # Quick test block
    import time
    print("Initializing Window Observer...")
    observer = UniversalWindowObserver()
    
    print("Switch windows over the next 5 seconds to test...")
    for _ in range(5):
        print(f"Active Window: '{observer.get_active_window_title()}'")
        time.sleep(1)