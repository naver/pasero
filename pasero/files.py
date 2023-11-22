# Pasero Copyright (c) 2023-present NAVER Corporation
# Please refer to the license file provided in the project.

import io
import os
import pickle
import logging
import numpy as np
import json
from typing import Optional, Iterator, Iterable


logger = logging.getLogger('files')


class File:
    """
    Iterates over text files or binary files containing numpy arrays.
    Also provides get_positions() which returns the starting position of each line and can be used to sample random
    lines.

    For example, to get a random array from a "numpy" file:

        ```    
        file = File.open(format='numpy', path=path)
        positions = file.get_positions()
        file.seek(np.random.choice(positions))
        print(next(file))
        ```
    
    The 'numpy' format corresponds to binary files starting with a pickled dict header and multiple numpy arrays
    serialized with array.tobytes(). We opted out of using np.save and np.load as they are excruciatingly slow.
    The header indicates the start position of all numpy arrays, their length, dimension and data type.
    """
    def __init__(self, path: str, store_files_under: Optional[int] = None):
        self._begin = self._position = 0
        self._path = path
        self._file = open(self._path, 'rb')
        
        if store_files_under:
            self._file.seek(0, io.SEEK_END)
            size = self._file.tell()
            self._file.seek(0)
            if size <= store_files_under:
                logger.info(f'storing file {path} in memory')
                content = self._file.read()
                self._file.close()
                self._file = io.BytesIO(content)        

    @classmethod
    def get_formats(cls) -> dict:
        return {'numpy': NumpyFile, 'txt': File, 'jsonl': JSONLFile}

    @classmethod
    def open(cls, *args, format: str = 'txt', **kwargs) -> 'File':
        cls = cls.get_formats().get(format, File)
        return cls(*args, **kwargs)

    def get_positions(self) -> tuple[np.ndarray, np.ndarray]:   # this is pretty costly
        pos = 0
        positions = []
        lengths = []
        for line in self:
            positions.append(pos)
            lengths.append(len(line))
            pos = self._file.tell()
        self._file.seek(self._begin)
        positions = np.array(positions, dtype=np.int64)
        lengths = np.array(lengths, dtype=np.int64)
        return positions, lengths

    def __next__(self) -> str:
        self.reopen()
        return next(self._file).strip().decode()

    def __iter__(self) -> Iterator[str]:
        while True:
            try:
                yield next(self)
            except StopIteration:
                break

    def close(self):
        if not isinstance(self._file, io.BytesIO) and not self._file.closed:
            self._position = self.tell()
            self._file.close()

    def reopen(self):
        if self._file.closed:
            self._file = open(self._path, 'rb')
            self._file.seek(self._position)

    def seek(self, offset, whence=0):
        self._position = offset
        self.reopen()
        self._file.seek(offset, whence)

    def tell(self) -> int:
        self.reopen()
        return self._file.tell()


class NumpyFile(File):
    def __init__(self, path: str, store_files_under: Optional[int] = None):
        super().__init__(path, store_files_under=store_files_under)
        header = pickle.load(self._file)
        self._dim = header['dim']
        self._dtype = header['dtype']
        self._dtype = np.dtype(self._dtype)
        self._itemsize = self._dim * self._dtype.itemsize
        positions = []
        lengths = []
        for pos, len_ in zip(header['positions'], header['lengths']):
            if pos > 0:  # skip empty positions
                positions.append(pos)
                lengths.append(len_)
        self._positions = np.array(positions, dtype=np.int64)
        self._lengths = np.array(lengths, dtype=np.int64)
        self._indices = np.arange(len(self._positions))
        self._index = 0

    @classmethod
    def build(
        cls,
        path: str,
        features: Iterable[np.ndarray],
        dtype: str = 'float16',
        num_feats: Optional[int] = None,
    ) -> 'NumpyFile':
        
        def write_header(file, positions, lengths, dim, dtype):
            dim = np.array(dim, dtype=np.int64)
            pickle.dump({'positions': positions, 'lengths': lengths, 'dim': dim, 'dtype': dtype}, file)

        if num_feats is None:  # can be larger than len(features)
            assert hasattr(features, '__len__')
            num_feats = len(features)

        dir = os.path.dirname(path)
        if dir:  # create parent directory if needed
            os.makedirs(dir, exist_ok=True)
        with open(path, 'wb') as file:
            dim = 0
            positions = np.zeros(num_feats, dtype=np.int64)
            lengths = np.zeros(num_feats, dtype=np.int64)
            write_header(file, positions, lengths, dim, dtype)

            for i, x in enumerate(features):
                x = x.astype(dtype)
                positions[i] = file.tell()
                lengths[i] = x.shape[0]
                dim = x.shape[1]
                bytes = x.tobytes()
                file.write(bytes)

            file.seek(0)
            write_header(file, positions, lengths, dim, dtype)

        return cls(path)

    def get_positions(self) -> tuple[np.ndarray, np.ndarray]:
        return self._indices, self._lengths

    def __next__(self) -> np.ndarray:
        self.reopen()
        if self._index >= len(self._positions):
            raise StopIteration
        length = self._lengths[self._index]
        size = length * self._itemsize
        x = self._file.read(size)
        x = np.frombuffer(x, dtype=self._dtype).copy()
        if self._dim > 1:
            x = x.reshape(length, self._dim)
        self._index += 1
        return x

    def close(self):
        if not isinstance(self._file, io.BytesIO) and not self._file.closed:
            self._file.close()

    def reopen(self):
        if self._file.closed:
            self._file = open(self._path, 'rb')
            self._file.seek(self._positions[self._index])

    def seek(self, offset, whence=0):
        self._index = offset
        self.reopen()
        self._file.seek(self._positions[offset], whence)

    def tell(self) -> int:
        return self._index


class JSONLFile(File):
    def __next__(self) -> str:
        line = super().__next__()
        return json.loads(line)
