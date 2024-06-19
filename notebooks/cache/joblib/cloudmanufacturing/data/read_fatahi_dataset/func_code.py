# first line: 25
@memory.cache
def read_fatahi_dataset(path_to_file):
    np.random.seed(42)
    """
    Fatahi Valilai, Omid. “Dataset for Logistics and Manufacturing
    Service Composition”. 17 Mar. 2021. Web. 9 June 2023.
    """
    workbook = load_workbook(filename=path_to_file, read_only=True)
    sheet_names = workbook.sheetnames

    res = []
    def process_sheet(sheet_name):
        return _read_sheet(path_to_file, sheet_name)
    res = Parallel(n_jobs=-1)(
        delayed(process_sheet)(sheet_name) for sheet_name in tqdm(sheet_names)
    )
    return res
