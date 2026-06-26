import numpy as np

def enforce_writeable_float_array_fast(data) -> np.ndarray:
    """
    যেকোনো ইনপুট ডেটাকে ফাস্ট, C-contiguous এবং writeable float64 NumPy অ্যারেতে কনভার্ট করে।
    মেমোরি লিক এবং ল্যাগ কমানোর জন্য এটি ইঞ্জিনের কোর ফাউন্ডেশন।
    """
    # ডেটা যদি অলরেডি নাম্পাই অ্যারে না হয়, তাহলে কনভার্ট করা
    if not isinstance(data, np.ndarray):
        arr = np.array(data, dtype=np.float64)
    else:
        # মেমোরি কপি না করে সরাসরি ভিউ নেওয়ার চেষ্টা
        arr = data.astype(np.float64, copy=False)
        
    # C-Contiguous নিশ্চিত করা (CPU Cache-এর ফাস্ট রিডিংয়ের জন্য)
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
        
    # অ্যারে যেন রাইটেবল হয়, যাতে পরবর্তী ক্যালকুলেশনে কোনো এরর না আসে
    if not arr.flags['WRITEABLE']:
        arr = arr.copy()
        arr.setflags(write=True)
        
    return arr

def get_rolling_window(arr: np.ndarray, window: int) -> np.ndarray:
    """
    স্লাইডিং উইন্ডো ক্যালকুলেশনের জন্য ফাস্টেস্ট মেথড (যেমন মুভিং এভারেজ বা রোলিং ভোলিটিলিটির জন্য)।
    এটি লুপ ছাড়া সরাসরি মেমোরি স্ট্রাইড (strides) ব্যবহার করে ভিউ রিটার্ন করে।
    """
    arr = enforce_writeable_float_array_fast(arr)
    
    # উইন্ডোর চেয়ে ডেটা ছোট হলে ফাঁকা অ্যারে রিটার্ন করবে
    if len(arr) < window:
        return np.array([], dtype=np.float64)
        
    shape = (arr.size - window + 1, window)
    strides = (arr.strides[0], arr.strides[0])
    
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

def replace_nan_with_zero(arr: np.ndarray) -> np.ndarray:
    """
    মার্কেটের মিসিং ডেটা (NaN) বা ইনফিনিটি (Inf) ভ্যালুকে জিরো (0.0) দিয়ে রিপ্লেস করে,
    যাতে ইঞ্জিনের কোনো ক্যালকুলেশন ক্র্যাশ না করে।
    """
    arr = enforce_writeable_float_array_fast(arr)
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return arr
