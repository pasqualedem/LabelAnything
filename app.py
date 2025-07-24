# from label_anything.demo.streamlit import main
from label_anything.demo.nicegui import main

if __name__ in {"__main__", "__mp_main__"}:
    main()